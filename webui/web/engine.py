"""Dataset-independent YOLO detection/segmentation inference on local media."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from functools import lru_cache

from library import now
from runtime import setup, setup_ultralytics, dump
from video_cache import PLAYBACK_VERSION, VideoEncoder, transcode

MAX_PIXELS = 25_000_000


class Engine:
    def __init__(self, library, output):
        self.library, self.output = library, Path(output)
        self.output.mkdir(exist_ok=True)
        self.models = {}
        self.lock = threading.Lock()
        self.initialized = False
        self.playback_tasks = {}
        self.playback_lock = threading.RLock()
        self.playback_serial = threading.Lock()

    def initialize(self):
        if self.initialized:
            return
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        setup()
        setup_ultralytics()
        import torch
        torch.set_num_threads(4)
        self.initialized = True

    def device(self, requested):
        import torch
        if requested not in {'auto', 'cpu', '0'}:
            raise ValueError('运行设备参数无效。')
        if requested == '0' and not torch.cuda.is_available():
            raise ValueError('当前没有可用的 CUDA 设备，请选择 CPU 或自动选择。')
        return '0' if requested == 'auto' and torch.cuda.is_available() else ('cpu' if requested == 'auto' else requested)

    def load(self, row, requested, progress=lambda message: None):
        progress('正在初始化推理环境…')
        self.initialize()
        from ultralytics import YOLO
        path = Path(row['path'])
        if not row['valid'] or not path.is_file():
            raise ValueError('模型记录已失效，请重新登记。')
        device = self.device(requested)
        if path.suffix.lower() == '.onnx':
            if not importlib.util.find_spec('onnxruntime') or not importlib.util.find_spec('onnx'):
                raise ValueError('当前 Python 环境未安装 onnx / onnxruntime，请安装 webui/requirements.txt 中的依赖。')
            import onnx
            import onnxruntime
            graph = onnx.load(str(path), load_external_data=False)
            metadata = {entry.key: entry.value for entry in graph.metadata_props}
            if metadata.get('task', row['task']) != row['task']:
                raise ValueError('ONNX 模型元数据中的任务与登记类型不一致，请按正确类型重新登记。')
            if device != 'cpu' and 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
                if requested == '0':
                    raise ValueError('ONNX Runtime 没有 CUDA provider，请选择 CPU 或自动选择。')
                device = 'cpu'
        signature = (row['id'], row['task'], device, path.stat().st_mtime_ns, path.stat().st_size)
        if signature not in self.models:
            progress(f'正在读取模型 {row["name"]}…')
            model = YOLO(str(path), task=row['task'])
            if model.task != row['task']:
                raise ValueError('模型实际任务与登记类型不同，请在模型库中按正确类型重新登记。')
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            # Bound cached weights so switching many models does not consume all VRAM.
            self.models.clear()
            self.models[signature] = (model, device, digest)
        else:
            progress('复用已加载的模型…')
        return self.models[signature]

    @staticmethod
    def read_image(path):
        from PIL import Image, ImageOps
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        try:
            with Image.open(path) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError('图片超过 2500 万像素，请缩小后测试。')
                image = ImageOps.exif_transpose(image).convert('RGBA')
                return Image.alpha_composite(Image.new('RGBA', image.size, 'white'), image).convert('RGB')
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError('无法读取图片，文件损坏或像素过多。') from exc

    def infer(self, model_info, image, parameters):
        import numpy as np
        model, device, digest = model_info
        started = time.perf_counter()
        result = model.predict(np.asarray(image)[:, :, ::-1].copy(),
                               conf=parameters['conf'], iou=parameters['iou'], imgsz=parameters['imgsz'],
                               device=device, max_det=300, verbose=False, save=False, retina_masks=True)[0]
        detections = []
        polygons = result.masks.xy if result.masks is not None else []
        for index, box in enumerate(result.boxes.data.cpu().tolist()):
            x1, y1, x2, y2, conf, cls = box[:6]
            cls = int(cls)
            color = tuple(50 + ((cls * factor + shift) % 185) for factor, shift in ((47, 40), (83, 95), (113, 25)))
            item = {'id': index + 1, 'class_id': cls, 'class_name': result.names[cls],
                    'confidence': round(conf, 6), 'bbox_xyxy': [round(v, 2) for v in (x1, y1, x2, y2)],
                    'color': '#' + ''.join(f'{c:02x}' for c in color),
                    'polygon': polygons[index].round(2).tolist() if index < len(polygons) else []}
            detections.append(item)
        detections.sort(key=lambda item: item['confidence'], reverse=True)
        return {'detections': detections, 'count': len(detections),
                'class_count': len({item['class_id'] for item in detections}),
                'average_confidence': round(sum(d['confidence'] for d in detections) / len(detections), 4) if detections else None,
                'width': image.width, 'height': image.height,
                'timing_ms': dict(inference_pure=round(result.speed.get('inference', 0), 2),
                                  preprocess=round(result.speed.get('preprocess', 0), 2),
                                  postprocess=round(result.speed.get('postprocess', 0), 2),
                                  total=round((time.perf_counter() - started) * 1000, 2)),
                'weights_sha256': digest, 'device': device}

    @staticmethod
    @lru_cache(maxsize=24)
    def label_font(size):
        from PIL import ImageFont
        font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/msyh.ttc'
        return ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default()

    @staticmethod
    def draw(image, report, parameters):
        from PIL import Image, ImageDraw, ImageFont
        canvas = image.copy().convert('RGBA')
        if parameters.get('show_masks', True):
            overlay = Image.new('RGBA', image.size)
            brush = ImageDraw.Draw(overlay)
            for d in report['detections']:
                if len(d['polygon']) >= 3:
                    brush.polygon([tuple(p) for p in d['polygon']], fill=d['color'] + '60')
            canvas = Image.alpha_composite(canvas, overlay)
        brush = ImageDraw.Draw(canvas)
        size = max(13, min(32, round(max(image.size) / 55)))
        font = Engine.label_font(size)
        for d in reversed(report['detections']):
            x1, y1, x2, y2 = d['bbox_xyxy']
            if parameters.get('show_boxes', True):
                brush.rectangle((x1, y1, x2, y2), outline=d['color'], width=max(2, image.width // 450))
            label = ('%s ' % d['class_name'] if parameters.get('show_names', True) else '')
            label += f"{d['confidence']:.2f}" if parameters.get('show_conf', True) else ''
            if label:
                width = brush.textlength(label, font=font) + 10
                x, y = max(0, min(x1, image.width - width)), max(0, y1 - size - 8)
                brush.rectangle((x, y, x + width, y + size + 7), fill=d['color'])
                brush.text((x + 5, y + 2), label, font=font, fill='white')
        return canvas.convert('RGB')

    def report_base(self, model, file, parameters):
        identifier = uuid.uuid4().hex
        folder = self.output / identifier
        folder.mkdir()
        return folder, {'id': identifier, 'created_at': now(), 'filename': file['name'],
                        'file_id': file['id'], 'model_key': model['id'], 'model': model['name'],
                        'task': model['task'], 'kind': file['kind'], 'parameters': parameters,
                        'original_url': f'/api/results/{identifier}/original.png',
                        'annotated_url': f'/api/results/{identifier}/annotated.png',
                        'json_url': f'/api/results/{identifier}/result.json'}

    def image(self, model, model_info, file, parameters):
        started = time.perf_counter()
        image = self.read_image(file['path'])
        folder, report = self.report_base(model, file, parameters)
        report.update(self.infer(model_info, image, parameters))
        image.save(folder / 'original.png')
        self.draw(image, report, parameters).save(folder / 'annotated.png')
        report['timing_ms']['total'] = round((time.perf_counter() - started) * 1000, 2)
        dump(folder / 'result.json', report)
        return report

    def video(self, model, model_info, file, parameters, stopped, progress):
        import cv2
        import numpy as np
        from PIL import Image
        capture = cv2.VideoCapture(file['path'])
        if not capture.isOpened():
            capture.release()
            raise ValueError('无法解码视频，请检查文件或转换为 MP4 / AVI。')
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if 0 < fps < 1000 else 25.0
        folder, report = self.report_base(model, file, parameters)
        encoders = []
        con = sqlite3.connect(folder / 'frames.sqlite3')
        con.execute('CREATE TABLE frames (frame INTEGER PRIMARY KEY, report TEXT NOT NULL)')
        frame, processed, infer_ms, detections_total = 0, 0, 0, 0
        report.update(total_frames=total, fps=fps, stride=parameters['stride'], processed_frames=0,
                      output_frames=0, status='running', phase='first_frame', playback_ready=False,
                      playback_version=PLAYBACK_VERSION,
                      source_signature=self.file_signature(file['path']),
                      model_signature=self.file_signature(model['path']), video_url=None,
                      original_video_url=None)
        started = time.perf_counter()
        last_publish = 0
        try:
            while not stopped():
                ok, bgr = capture.read()
                if not ok:
                    break
                if bgr.shape[0] * bgr.shape[1] > MAX_PIXELS:
                    raise ValueError('视频单帧超过 2500 万像素。')
                frame += 1
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if not encoders:
                    progress(report, 0, total)
                    encoders = [VideoEncoder(folder, 'annotated', image.width, image.height, fps)]
                    encoders.append(VideoEncoder(folder, 'original', image.width, image.height, fps))
                if (frame - 1) % parameters['stride'] == 0:
                    single = self.infer(model_info, image, parameters)
                    single.update(frame_index=frame - 1, time_seconds=round((frame - 1) / fps, 3))
                    con.execute('INSERT INTO frames VALUES (?,?)', (frame - 1, json.dumps(single, ensure_ascii=False)))
                    processed += 1
                    infer_ms += single['timing_ms']['inference_pure']
                    detections_total += single['count']
                    report.update(single, processed_frames=processed, detections_total=detections_total,
                                  mean_inference_ms=round(infer_ms / processed, 2))
                # Sampling reduces inference work; the output still contains every
                # source frame at its original rate, holding the latest prediction.
                annotated = self.draw(image, single, parameters)
                encoders[0].write(cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
                encoders[1].write(bgr)
                report.update(phase='precomputing', output_frames=frame,
                              elapsed_seconds=round(time.perf_counter() - started, 2))
                if time.monotonic() - last_publish > 1:
                    con.commit()
                    self.save_preview(folder, image, annotated)
                    dump(folder / 'result.json', report)
                    progress(report, frame, total)
                    last_publish = time.monotonic()
            if not processed:
                if stopped():
                    report.update(status='cancelled')
                else:
                    raise ValueError('视频没有可解码的画面。')
            else:
                self.save_preview(folder, image, annotated)
                report.update(phase='finalizing')
                progress(report, frame, total)
                for encoder in encoders:
                    encoder.finish()
                report.update(playback_ready=True,
                              video_url=f'/api/results/{report["id"]}/annotated.mp4',
                              original_video_url=f'/api/results/{report["id"]}/original.mp4',
                              duration_seconds=round(frame / fps, 6))
            report.update(status='cancelled' if stopped() else 'completed', processed_frames=processed)
            report['phase'] = 'ready'
            dump(folder / 'result.json', report)
            return report
        finally:
            con.commit()
            con.close()
            capture.release()
            for encoder in encoders:
                encoder.abort()

    @staticmethod
    def save_preview(folder, image, annotated):
        for name, value in [('original', image), ('annotated', annotated)]:
            temp = folder / (name + '.tmp.png')
            value.save(temp, compress_level=1)
            temp.replace(folder / (name + '.png'))

    @staticmethod
    def file_signature(path):
        path = Path(path)
        stat = path.stat()
        return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

    def cached_video(self, model, file, parameters):
        with self.library.connect() as con:
            rows = con.execute('''SELECT result_id FROM outcomes WHERE model_id=? AND file_id=?
                AND status='completed' ORDER BY created_at DESC,rowid DESC LIMIT 20''',
                               (model['id'], file['id'])).fetchall()
        for row in rows:
            folder = self.output / row['result_id']
            try:
                report = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
                if (report.get('playback_version') == PLAYBACK_VERSION
                    and report.get('playback_ready') and report.get('parameters') == parameters
                    and report.get('source_signature') == self.file_signature(file['path'])
                    and report.get('model_signature') == self.file_signature(model['path'])
                    and all((folder / name).is_file() for name in ('original.mp4', 'annotated.mp4'))):
                    return dict(report, reused=True)
            except (OSError, ValueError, KeyError):
                continue
        return None

    def prepare_playback(self, result_id):
        folder = self.output / result_id
        report = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        if report.get('kind') != 'video':
            raise ValueError('该结果不是视频。')
        if report.get('playback_ready') and (folder / 'annotated.mp4').is_file():
            return {'status': 'completed', 'report': report, 'message': '视频缓存已就绪'}
        with self.playback_lock:
            task = self.playback_tasks.get(result_id)
            if task and task['status'] != 'failed':
                return dict(task)
            if not (folder / 'annotated.avi').is_file():
                raise ValueError('视频还在预计算，请等待测试完成。')
            self.playback_tasks[result_id] = {'status': 'running', 'message': '正在转换历史视频缓存，无需重新推理…'}
            threading.Thread(target=self._upgrade_playback, args=(result_id, report), daemon=True).start()
            return dict(self.playback_tasks[result_id])

    def playback_status(self, result_id):
        with self.playback_lock:
            return dict(self.playback_tasks.get(result_id) or self.prepare_playback(result_id))

    def _upgrade_playback(self, result_id, report):
        folder = self.output / result_id
        try:
            with self.playback_serial:
                fps = report.get('fps') or 25
                covered = report['processed_frames'] * report.get('stride', 1)
                covered = min(covered, report.get('total_frames') or covered)
                duration = covered / fps
                if not (folder / 'annotated.mp4').is_file():
                    transcode(folder / 'annotated.avi', folder / 'annotated.mp4', fps, duration)
                file = self.library.get('files', report['file_id'])
                if not (folder / 'original.mp4').is_file():
                    transcode(file['path'], folder / 'original.mp4', fps, duration)
                report.update(playback_ready=True, playback_version=PLAYBACK_VERSION,
                              output_frames=round(duration * fps), duration_seconds=duration,
                              original_video_url=f'/api/results/{result_id}/original.mp4',
                              video_url=f'/api/results/{result_id}/annotated.mp4')
                # Admit a legacy result to the inference cache only when its
                # original model digest and cataloged media still match.
                try:
                    from datetime import datetime
                    model = self.library.get('models', report['model_key'])
                    stat = Path(file['path']).stat()
                    created = datetime.fromisoformat(report['created_at']).timestamp()
                    with Path(model['path']).open('rb') as stream:
                        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if (digest == report.get('weights_sha256') and stat.st_mtime <= created
                        and stat.st_size == file['size'] and abs(stat.st_mtime - file['mtime']) < .000001):
                        report['source_signature'] = self.file_signature(file['path'])
                        report['model_signature'] = self.file_signature(model['path'])
                except (KeyError, ValueError, OSError):
                    pass
                dump(folder / 'result.json', report)
            with self.playback_lock:
                self.playback_tasks[result_id] = {'status': 'completed', 'report': report, 'message': '视频缓存已就绪'}
        except Exception as exc:
            with self.playback_lock:
                self.playback_tasks[result_id] = {'status': 'failed', 'message': str(exc)}

    def frame(self, result_id, index):
        folder = self.output / result_id
        con = sqlite3.connect(folder / 'frames.sqlite3')
        try:
            row = con.execute('SELECT report FROM frames WHERE frame<=? ORDER BY frame DESC LIMIT 1', (index,)).fetchone()
            if not row:
                row = con.execute('SELECT report FROM frames ORDER BY frame LIMIT 1').fetchone()
        finally:
            con.close()
        if not row:
            raise ValueError('尚无已测试的视频帧。')
        return json.loads(row[0])

    def frame_image(self, result_id, index):
        import cv2
        from PIL import Image
        report = json.loads((self.output / result_id / 'result.json').read_text(encoding='utf-8'))
        file = self.library.get('files', report['file_id'])
        capture = cv2.VideoCapture(file['path'])
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok:
                raise ValueError('该视频帧已不可读取。')
            stream = io.BytesIO()
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(stream, 'JPEG', quality=90)
            return stream.getvalue()
        finally:
            capture.release()
