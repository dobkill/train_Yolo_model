"""One worker per workstation, with persistent progress and cooperative cancellation."""
import copy
import math
from pathlib import Path
import threading
import time
import traceback
import uuid

from library import now, yaml_config
from runtime import dump


def parameters(body):
    value = dict(conf=float(body.get('conf', .35)), iou=float(body.get('iou', .5)),
                 imgsz=int(body.get('imgsz', 640)), device=str(body.get('device', 'auto')),
                 stride=int(body.get('stride', 1)))
    if not all(math.isfinite(value[k]) and 0 <= value[k] <= 1 for k in ('conf', 'iou')):
        raise ValueError('置信度和 IoU 阈值必须在 0–1 之间。')
    if value['imgsz'] not in {320, 640, 960, 1280} or not 1 <= value['stride'] <= 60:
        raise ValueError('输入尺寸或视频帧间隔无效。')
    if value['device'] not in {'auto', 'cpu', '0'}:
        raise ValueError('运行设备无效。')
    for name in ('show_names', 'show_conf', 'show_boxes', 'show_masks'):
        value[name] = bool(body.get(name, True))
    return value


class Jobs:
    def __init__(self, library, engine):
        self.library, self.engine = library, engine
        self.lock = threading.RLock()
        self.active = None
        self.stop_event = threading.Event()
        self.last_checkpoint = 0

    def start(self, body):
        params = parameters(body.get('parameters', {}))
        model = self.library.get('models', body.get('model_id', ''))
        dataset = self.library.get('datasets', body.get('dataset_id', ''))
        if not model['valid'] or not dataset['valid']:
            raise ValueError('所选记录已失效，请在库中重新检查或登记。')
        if not Path(model['path']).is_file() or not Path(dataset['path']).exists():
            raise ValueError('源文件已移动或删除，请重新检查记录。')
        mode = body.get('mode', 'preview')
        if mode not in {'preview', 'evaluate'}:
            raise ValueError('测试模式无效。')
        kind = body.get('kind', 'image')
        if kind not in {'image', 'video'}:
            raise ValueError('数据类型无效。')
        if mode == 'evaluate':
            path = Path(dataset['path'])
            if kind != 'image' or path.suffix.lower() not in {'.yaml', '.yml'}:
                raise ValueError('定量评测需要登记包含标注的 YOLO 数据集 YAML，仅支持图片。')
            config, _ = yaml_config(path)
            if not (config.get('test') or config.get('val')):
                raise ValueError('定量评测需要有效的 test 或 val 图片路径。')
        files = self.library.dataset_files(dataset['id'], kind)
        if body.get('file_id'):
            files = [file for file in files if file['id'] == body['file_id']]
        if not files:
            raise ValueError('所选数据集中没有可测试的文件。')
        with self.lock:
            if self.active and self.active['status'] in {'running', 'queued', 'stopping'}:
                raise ValueError('已有测试正在运行，请等待结束或点击停止。')
            self.stop_event.clear()
            self.active = {'id': uuid.uuid4().hex, 'status': 'queued', 'created_at': now(),
                           'model_id': model['id'], 'model_name': model['name'], 'task': model['task'],
                           'dataset_id': dataset['id'], 'dataset_name': dataset['name'],
                           'mode': mode, 'kind': kind, 'parameters': params, 'total': len(files),
                           'completed': 0, 'failed': 0, 'progress': 0, 'message': '正在加载模型…',
                           'phase': 'loading_model', 'cached': 0,
                           'current_file': files[0]['name'], 'current_file_id': files[0]['id'],
                           'results': [], 'errors': [], 'latest_result': None}
            self.library.save_job(self.active)
            self.library.set_settings({'model_id': model['id'], 'dataset_id': dataset['id'],
                                       'task': model['task'], 'kind': kind, 'parameters': params,
                                       'last_test_at': self.active['created_at']})
            snapshot = copy.deepcopy(self.active)
            threading.Thread(target=self.run, args=(model, dataset, files), daemon=True).start()
            return snapshot

    def update(self, **values):
        with self.lock:
            self.active.update(values)
            if time.monotonic() - self.last_checkpoint >= 1:
                self.library.save_job(self.active)
                self.last_checkpoint = time.monotonic()

    def get(self, identifier):
        with self.lock:
            if self.active and identifier == self.active['id']:
                return copy.deepcopy(self.active)
        return self.library.job(identifier)

    def cancel(self, identifier):
        with self.lock:
            if not self.active or identifier != self.active['id']:
                raise ValueError('该测试已结束。')
            if self.active['status'] in {'running', 'queued', 'stopping'}:
                self.stop_event.set()
                self.active.update(status='stopping', message='正在停止，等待当前推理完成…')
            return copy.deepcopy(self.active)

    def run(self, model, dataset, files):
        job = self.active
        try:
            with self.engine.lock:
                info = None
                if self.stop_event.is_set():
                    self.update(status='cancelled', message='测试已停止。')
                    return
                self.update(status='running')
                if job['mode'] == 'evaluate':
                    info = self.engine.load(model, job['parameters']['device'],
                                            lambda message: self.update(phase='loading_model', message=message))
                    self.evaluate(model, dataset, info)
                    return
                for position, file in enumerate(files):
                    if self.stop_event.is_set():
                        break
                    self.update(current_file=file['name'], current_file_id=file['id'], message=f'正在测试 {file["name"]}')
                    try:
                        cached = self.engine.cached_video(model, file, job['parameters']) if file['kind'] == 'video' else None
                        if cached:
                            report = cached
                            self.update(phase='cached', message=f'{file["name"]} · 已复用预计算结果')
                            job['cached'] += 1
                        else:
                            if info is None:
                                info = self.engine.load(model, job['parameters']['device'],
                                                        lambda message: self.update(phase='loading_model', message=message))
                            self.update(phase='precomputing', message=f'正在预计算 {file["name"]}')
                        if file['kind'] == 'image':
                            report = self.engine.image(model, info, file, job['parameters'])
                        elif not cached:
                            def progress(report, frame, total):
                                phase = report.get('phase', 'precomputing')
                                fraction = min(.98, frame / total * .98) if total else 0
                                message = ('首帧初始化与预热…' if phase == 'first_frame' else
                                           '推理完成，正在整理 MP4 播放缓存…' if phase == 'finalizing' else
                                           f'正在预计算 · {report["processed_frames"]}/{total or "未知"} 帧')
                                self.update(latest_result=copy.deepcopy(report),
                                            progress=round((position + fraction) / len(files) * 100, 1),
                                            phase=phase, message=message)
                            report = self.engine.video(model, info, file, job['parameters'], self.stop_event.is_set, progress)
                        with self.lock:
                            job['results'].append({'file_id': file['id'], 'result_id': report['id'], 'kind': file['kind']})
                            job['latest_result'] = report
                            job['completed'] += int(report.get('status') != 'cancelled')
                        self.library.outcome(job, file['id'], report['id'], report.get('status', 'completed'))
                    except Exception as exc:
                        traceback.print_exc()
                        with self.lock:
                            job['errors'].append({'file': file['name'], 'error': str(exc)})
                            job['failed'] += 1
                        self.library.outcome(job, file['id'], None, 'failed')
                    self.update(progress=round((position + 1) / len(files) * 100, 1))
                    self.library.save_job(job)
                status = 'cancelled' if self.stop_event.is_set() else ('completed_with_errors' if job['failed'] else 'completed')
                self.update(status=status, phase='ready', message=f'已停止 · 完成 {job["completed"]} 个' if status == 'cancelled'
                            else f'测试完成 · 成功 {job["completed"]}，失败 {job["failed"]}'
                            + (f' · 复用缓存 {job["cached"]} 个' if job['cached'] else ''))
        except Exception as exc:
            traceback.print_exc()
            self.update(status='failed', error=str(exc), message=str(exc))
        finally:
            self.update(finished_at=now())
            self.library.save_job(job)

    def evaluate(self, model, dataset, info):
        job = self.active
        config, sources = yaml_config(Path(dataset['path']))
        split = 'test' if sources.get('test') else 'val'
        folder = self.engine.output / job['id']
        folder.mkdir(exist_ok=True)
        import yaml
        config_path = folder / 'dataset.yaml'
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
        yolo, device, _ = info

        def on_batch(validator):
            if self.stop_event.is_set():
                raise InterruptedError('定量评测已停止。')
            batches = max(1, len(validator.dataloader))
            self.update(progress=round((validator.batch_i + 1) / batches * 100, 1),
                        message=f'正在评测 {split} · 批次 {validator.batch_i + 1}/{batches}')

        yolo.add_callback('on_val_batch_end', on_batch)
        try:
            from ultralytics.data.utils import img2label_paths
            images = self.library.dataset_files(dataset['id'], 'image')
            label_paths = img2label_paths([file['path'] for file in images])
            if not any(Path(label).is_file() for label in label_paths):
                raise ValueError('评测图片没有对应的 YOLO 标注文件，请检查 images / labels 结构。')
            expected_names = config['names']
            expected_names = dict(enumerate(expected_names)) if isinstance(expected_names, list) else {int(k): str(v) for k, v in expected_names.items()}
            if Path(model['path']).suffix.lower() == '.pt' and dict(yolo.names) != expected_names:
                raise ValueError('数据集类别定义与模型类别不一致，请选择与该模型配套的评测 YAML。')
            metrics = yolo.val(data=str(config_path), split=split, device=device,
                               imgsz=job['parameters']['imgsz'], conf=job['parameters']['conf'],
                               iou=job['parameters']['iou'], batch=1, workers=0, plots=False,
                               save_json=False, project=str(folder), name='validation', exist_ok=True, verbose=False)
            values = {str(k): float(v) for k, v in metrics.results_dict.items()}
            dump(folder / 'metrics.json', values)
            self.update(status='completed', progress=100, metrics=values, split=split,
                        completed=job['total'], message='定量评测完成', metrics_url=f'/api/results/{job["id"]}/metrics.json')
        except InterruptedError:
            self.update(status='cancelled', message='定量评测已停止。')
        finally:
            yolo.callbacks['on_val_batch_end'].remove(on_batch)
