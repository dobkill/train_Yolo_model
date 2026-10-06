"""Local image detector web UI. Supports model selection, pure-inference timing,
GPU utilization/memory stats and folder uploads. Uses only the local Python env."""
import argparse
from datetime import datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB.parent))
from runtime import ROOT, dump, setup, setup_ultralytics

MAX_UPLOAD = 25 * 1024 * 1024
MAX_PIXELS = 25_000_000
APP_ID = "arad-local-detector-v1"


class RequestError(Exception):
    def __init__(self, message, status=400):
        self.status = status
        super().__init__(message)


# key -> {path, label, metrics_path or None}; entries need a .pt file on disk.
def build_registry():
    registry = {}
    primary = ROOT / "runs/named116_yolo11n/weights/best.pt"
    if primary.is_file():
        registry["named116_best"] = {"path": primary, "label": "YOLO11n · 116 类 · best.pt",
                                     "metrics_path": ROOT / "runs/named116_yolo11n/test_metrics.json"}
    last = ROOT / "runs/named116_yolo11n/weights/last.pt"
    if last.is_file():
        registry["named116_last"] = {"path": last, "label": "YOLO11n · 116 类 · last.pt", "metrics_path": None}
    for candidate in sorted((ROOT / "pretrained").glob("*.pt")):
        registry[f"pretrained_{candidate.stem}"] = {"path": candidate, "label": f"{candidate.stem} · 官方预训练(COCO 80 类)",
                                                    "metrics_path": None}
    if not registry:
        raise FileNotFoundError("Work/train 下没有可用的 .pt 权重(runs 与 pretrained 均为空)。")
    return registry


class Detector:
    def __init__(self):
        os.environ["YOLO_AUTOINSTALL"] = "false"
        setup()
        setup_ultralytics()
        import torch
        from PIL import Image, ImageFont
        from ultralytics import YOLO
        from ultralytics.utils import USER_CONFIG_DIR
        import numpy as np

        self.Image = Image
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        torch.set_num_threads(4)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if self.device == 0 else "CPU"
        self.registry = build_registry()
        self.active = None
        self.lock = threading.Lock()
        self.models = {}
        self.font_path = str(USER_CONFIG_DIR / "Arial.Unicode.ttf")
        ImageFont.truetype(self.font_path, 14)
        (WEB / "results").mkdir(exist_ok=True)
        # Keep dataset class list for detection models trained on it.
        # Dataset may live under Work/ (legacy) or Work/Datas/ (current).
        candidates = [ROOT.parent / "dataset_named_1000/classes.txt",
                      ROOT.parent / "Datas/dataset_named_1000/classes.txt"]
        classes_file = next((path for path in candidates if path.is_file()), None)
        if classes_file is None:
            raise FileNotFoundError("找不到 dataset_named_1000/classes.txt（Work 与 Work/Datas 下均不存在）。")
        self.dataset_classes = classes_file.read_text(encoding="utf-8").splitlines()
        self.load("named116_best" if "named116_best" in self.registry else next(iter(self.registry)))
        self.load_examples()

    def load_examples(self):
        manifest = ROOT / "runs/named116_yolo11n/predictions/manifest.json"
        manifest = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else []
        self.examples = []
        for row in manifest[:4]:
            path = Path(row["source"])
            if not path.is_file():
                # Dataset was moved under Work/Datas after training; remap legacy paths.
                parts = path.parts
                if "dataset_named_1000" in parts:
                    index = parts.index("dataset_named_1000")
                    path = ROOT.parent / "Datas" / Path(*parts[index:])
            if path.is_file():
                self.examples.append(path)

    def load(self, key):
        """Load one registry entry; returns model info dict. Thread-safe via self.lock."""
        from ultralytics import YOLO
        entry = self.registry.get(key)
        if entry is None:
            raise RequestError(f"未知模型: {key}", 404)
        with self.lock:
            info = self.models.get(key)
            if info is None:
                weights = entry["path"]
                weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
                model = YOLO(str(weights), task="detect")
                assert model.task == "detect", "本页面只支持检测模型"
                if len(model.names) == len(self.dataset_classes) and model.names == dict(enumerate(self.dataset_classes)):
                    classes_note = "116 类数据集训练"
                else:
                    classes_note = "类别与数据集不一致(COCO/其他)"
                metrics = None
                if entry["metrics_path"] and entry["metrics_path"].is_file():
                    metrics = json.loads(entry["metrics_path"].read_text(encoding="utf-8"))
                    assert metrics["weights_sha256"] == weights_sha256, "权重与测试报告不匹配"
                import numpy as np
                import torch
                model.predict(np.zeros((480, 640, 3), dtype=np.uint8), device=self.device,
                              imgsz=640, verbose=False, save=False)
                info = {"model": model, "weights": weights, "weights_sha256": weights_sha256,
                        "classes_note": classes_note, "metrics": metrics,
                        "classes": len(model.names), "names": model.names}
                self.models[key] = info
            self.active = key
            return info

    def info(self):
        entry_info = self.models[self.active]
        metrics = entry_info["metrics"]
        return {"app": APP_ID, "ready": True, "active": self.active,
                "models": [{"key": key, "label": item["label"], "loaded": key in self.models,
                            "is_active": key == self.active}
                           for key, item in self.registry.items()],
                "model": self.registry[self.active]["label"], "classes": entry_info["classes"],
                "classes_note": entry_info["classes_note"],
                "device": self.device_name, "weights": str(self.registry[self.active]["path"].relative_to(ROOT)),
                "weights_sha256": entry_info["weights_sha256"],
                "test_map50": metrics["metrics"]["metrics/mAP50(B)"] if metrics else None,
                "test_map50_95": metrics["metrics"]["metrics/mAP50-95(B)"] if metrics else None,
                "test_scope": "合成测试集" if metrics else None,
                "max_upload_mb": MAX_UPLOAD // 1024**2,
                "examples": [{"id": i, "name": p.name, "url": f"/api/examples/{i}"}
                             for i, p in enumerate(self.examples)]}

    def gpu_stats(self):
        import torch
        if self.device == "cpu" or not torch.cuda.is_available():
            return {"available": False, "util_pct": None, "mem_used_mb": None, "mem_total_mb": None,
                    "mem_pct": None}
        used, total = torch.cuda.memory_allocated() / 1048576, torch.cuda.memory_reserved() / 1048576
        try:
            import pynvml
        except ImportError:
            pynvml = None
        util = total_pct = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_pct = memory_info.used / memory_info.total * 100
                used = memory_info.used / 1048576
                total = memory_info.total / 1048576
                pynvml.nvmlShutdown()
            except Exception:
                util = total_pct = None
        return {"available": True, "util_pct": util, "mem_used_mb": round(used, 1),
                "mem_total_mb": round(total, 1),
                "mem_pct": round(total_pct, 1) if total_pct is not None else None}

    def predict(self, payload, filename, conf, iou, imgsz):
        from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
        import numpy as np

        started = time.perf_counter()
        try:
            with Image.open(io.BytesIO(payload)) as opened:
                if opened.format not in {"PNG", "JPEG", "WEBP", "BMP"}:
                    raise RequestError("请选择 PNG、JPG、WEBP 或 BMP 图片。")
                if opened.width * opened.height > MAX_PIXELS:
                    raise RequestError("图片超过 2500 万像素，请缩小后重试。", 413)
                image = ImageOps.exif_transpose(opened).convert("RGBA")
                background = Image.new("RGBA", image.size, "white")
                image = Image.alpha_composite(background, image).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RequestError("无法读取这张图片，请确认图片完整且格式受支持。") from exc
        except Image.DecompressionBombError as exc:
            raise RequestError("图片像素过多，请缩小后重试。", 413) from exc
        if not self.lock.acquire(blocking=False):
            raise RequestError("模型正在处理另一张图片，请稍后再试。", 429)
        try:
            preprocess_started = time.perf_counter()
            bgr = np.asarray(image)[:, :, ::-1].copy()  # Ultralytics numpy input is BGR.
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
            gpu_before = self.gpu_stats()
            inference_started = time.perf_counter()
            result = self.model_predict(bgr, conf, iou, imgsz)
            inference_wall_ms = (time.perf_counter() - inference_started) * 1000
            rows = result.boxes.data.detach().cpu().tolist()
            postprocess_ms = (time.perf_counter() - inference_started) * 1000 - inference_wall_ms
            gpu_after = self.gpu_stats()
        finally:
            self.lock.release()
        # Ultralytics speed dict is ms per batch (preprocess/inference/postprocess).
        speeds = result.speed
        rows.sort(key=lambda row: row[4], reverse=True)
        w, h = image.size
        names = self.models[self.active]["names"]
        detections = []
        for index, (x1, y1, x2, y2, confidence, cls) in enumerate(rows):
            cls = int(cls)
            color = tuple(65 + ((cls * factor + shift) % 160) for factor, shift in ((47, 50), (83, 95), (113, 10)))
            detections.append({"id": index + 1, "class_id": cls, "class_name": names[cls],
                 "confidence": round(confidence, 6),
                 "bbox_xyxy": [round(max(0, min(limit, v)), 2)
                               for v, limit in zip((x1, y1, x2, y2), (w, h, w, h))],
                 "color": "#" + "".join(f"{c:02x}" for c in color)})
        total_ms = (time.perf_counter() - started) * 1000
        identifier = uuid.uuid4().hex
        folder = WEB / "results" / identifier
        folder.mkdir()
        image.save(folder / "original.png")
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        font_size = max(13, min(36, round(max(w, h) / 45)))
        font = ImageFont.truetype(self.font_path, font_size)
        line_width = max(2, round(max(w, h) / 450))
        draw_started = time.perf_counter()
        for item in reversed(detections):
            x1, y1, x2, y2 = item["bbox_xyxy"]
            color = item["color"]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            label = f'{item["class_name"]} {item["confidence"]:.0%}'
            tw = draw.textlength(label, font=font) + 10
            tx, ty = max(0, min(x1, w - tw)), max(0, y1 - font_size - 10)
            draw.rectangle((tx, ty, tx + tw, ty + font_size + 8), fill=color)
            draw.text((tx + 5, ty + 3), label, font=font, fill="#07111d", stroke_width=0)
        annotated.save(folder / "annotated.png")
        draw_ms = (time.perf_counter() - draw_started) * 1000
        report = {"id": identifier, "created_at": datetime.now().isoformat(timespec="seconds"),
             "filename": filename, "width": w, "height": h, "model": self.registry[self.active]["label"],
             "model_key": self.active, "device": self.device_name,
             "weights_sha256": self.models[self.active]["weights_sha256"],
             "input_sha256": hashlib.sha256(payload).hexdigest(),
             "bbox_type": "predicted_visible_bbox", "parameters": {"confidence": conf, "iou": iou, "imgsz": imgsz},
             "detections": detections, "count": len(detections),
             "class_count": len({d["class_id"] for d in detections}),
             "timing_ms": {"preprocess": round(preprocess_ms, 2),
                           "inference_pure": round(speeds.get("inference", inference_wall_ms), 2),
                           "postprocess": round(speeds.get("postprocess", postprocess_ms), 2),
                           "drawing": round(draw_ms, 2),
                           "total": round(total_ms, 1)},
             "gpu": gpu_after,
             "original_url": f"/api/results/{identifier}/original.png",
             "annotated_url": f"/api/results/{identifier}/annotated.png",
             "json_url": f"/api/results/{identifier}/result.json"}
        dump(folder / "result.json", report)
        return report

    def model_predict(self, bgr, conf, iou, imgsz):
        return self.models[self.active]["model"].predict(
            bgr, device=self.device, conf=conf, iou=iou, imgsz=imgsz, max_det=300,
            verbose=False, save=False)[0]


class Handler(BaseHTTPRequestHandler):
    server_version = "AradDetector/1.1"

    def reply(self, content, status=200, mime="application/json; charset=utf-8", download=None):
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def file(self, path, download=None):
        if not path.is_file():
            raise RequestError("文件不存在。", 404)
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix in {".html", ".css", ".js", ".json"}:
            mime += "; charset=utf-8"
        self.reply(path.read_bytes(), mime=mime, download=download)

    def local_request(self):
        host = urlsplit("http://" + self.headers.get("Host", "")).hostname
        if host not in {"127.0.0.1", "localhost"}:
            raise RequestError("服务仅接受本机访问。", 403)
        origin = self.headers.get("Origin")
        if origin and (urlsplit(origin).hostname not in {"127.0.0.1", "localhost"}
                       or urlsplit(origin).port != self.server.server_port):
            raise RequestError("请从本地识别网页提交图片。", 403)

    def do_GET(self):
        try:
            self.local_request()
            path = unquote(urlsplit(self.path).path)
            if path == "/api/health":
                return self.reply(self.server.detector.info())
            if path == "/api/gpu":
                return self.reply(self.server.detector.gpu_stats())
            if path in {"/", "/index.html", "/app.css", "/app.js", "/favicon.svg"}:
                return self.file(WEB / "static" / ("index.html" if path == "/" else path.lstrip("/")))
            match = re.fullmatch(r"/api/examples/([0-3])", path)
            if match:
                return self.file(self.server.detector.examples[int(match[1])])
            match = re.fullmatch(r"/api/results/([0-9a-f]{32})/(original\.png|annotated\.png|result\.json)", path)
            if match:
                download = match[2] if "download" in parse_qs(urlsplit(self.path).query) else None
                return self.file(WEB / "results" / match[1] / match[2], download)
            raise RequestError("页面不存在。", 404)
        except RequestError as exc:
            self.reply({"error": str(exc)}, exc.status)

    def do_POST(self):
        try:
            if urlsplit(self.path).path == "/api/model":
                self.local_request()
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                key = str(body.get("key", ""))
                if not key:
                    raise RequestError("缺少模型标识 key。")
                self.server.detector.load(key)
                return self.reply(self.server.detector.info())
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_UPLOAD:
                self.close_connection = True
                raise RequestError("请选择不超过 25 MB 的图片。", 413)
            # Consume the bounded request body before a validation error. Closing
            # a Windows socket with unread upload bytes may reset the response.
            self.connection.settimeout(60)
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise RequestError("图片上传不完整，请重试。")
            self.local_request()
            if urlsplit(self.path).path != "/api/predict":
                raise RequestError("接口不存在。", 404)
            query = parse_qs(urlsplit(self.path).query)
            conf = float(query.get("conf", ["0.25"])[0])
            iou = float(query.get("iou", ["0.7"])[0])
            imgsz = int(query.get("imgsz", ["640"])[0])
            if not math.isfinite(conf) or not 0.01 <= conf <= 0.95:
                raise RequestError("置信度应在 0.01–0.95 之间。")
            if not math.isfinite(iou) or not 0.1 <= iou <= 0.95 or imgsz not in {640, 960, 1280}:
                raise RequestError("识别参数无效。")
            filename = query.get("filename", ["image"])[0][:200]
            result = self.server.detector.predict(payload, filename, conf, iou, imgsz)
            self.reply(result)
        except RequestError as exc:
            self.reply({"error": str(exc)}, exc.status)
        except (ValueError, TimeoutError) as exc:
            self.reply({"error": "请求参数或上传数据无效，请重新选择图片。"}, 400)
        except Exception:
            traceback.print_exc()
            self.reply({"error": "识别失败，请重试；详细原因已记录到本地服务日志。"}, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print("Loading model registry...", flush=True)
    try:
        server.detector = Detector()
        print(f"READY http://127.0.0.1:{args.port} ({server.detector.device_name})", flush=True)
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
