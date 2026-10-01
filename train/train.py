"""Train the existing 116-class visible-box dataset, then evaluate held-out test."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

from runtime import ROOT, dump, resolved, setup, setup_ultralytics

setup()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preflight(data_path, output):
    import torch
    import torchvision
    import ultralytics
    import yaml
    from torchvision.ops import nms

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to silently train on CPU")
    # Also exercise the compiled torchvision CUDA extension.
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
    assert nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5).tolist() == [0]
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset = data_path.parent
    validation = json.loads((dataset / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("valid"):
        raise ValueError("Existing dataset validation did not pass")
    names = data["names"]
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
    assert names == (dataset / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert len(names) == 116
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "torch": torch.__version__,
              "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
              "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "compute_capability": torch.cuda.get_device_capability(0),
              "data_yaml": str(data_path), "data_yaml_sha256": sha256(data_path),
              "class_names": names, "bbox_target": "bbox_visible", "splits": {},
              "dataset_validation_sha256": sha256(dataset / "validation.json"),
              "limitations": json.loads((dataset / "class_counts.json").read_text(encoding="utf-8"))}
    seen = set()
    for split, quota in (("train", 800), ("val", 100), ("test", 100)):
        print(f"Checking existing {split} detection labels...", flush=True)
        images_dir = Path(data["path"]) / data[split]
        labels_dir = images_dir.parent.parent / "labels" / split
        images = sorted(images_dir.glob("*.png"))
        assert images, images_dir
        count = Counter()
        negatives = 0
        for image in images:
            assert image.stem not in seen, f"Repeated scene across splits: {image}"
            seen.add(image.stem)
            lines = (labels_dir / (image.stem + ".txt")).read_text(encoding="utf-8").splitlines()
            negatives += not lines
            image_classes = set()
            for line in lines:
                values = line.split()
                assert len(values) == 5, f"Not detection xywh: {image}"
                cls = int(values[0])
                x, y, w, h = map(float, values[1:])
                assert 0 <= cls < len(names) and cls not in image_classes
                assert all(math.isfinite(v) for v in (x, y, w, h))
                assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                assert min(x-w/2, y-h/2) >= -1e-5 and max(x+w/2, y+h/2) <= 1+1e-5
                count[cls] += 1
                image_classes.add(cls)
        assert all(count[i] == quota for i in range(len(names))), (split, count)
        report["splits"][split] = {"images": len(images), "instances": sum(count.values()),
                                   "negative_images": negatives, "per_class_instances": quota}
        print(split, report["splits"][split], flush=True)
    report["free_disk_bytes"] = shutil.disk_usage(ROOT).free
    if report["free_disk_bytes"] < 500 * 1024**2:
        raise RuntimeError("Need at least 500 MiB free for logs, label caches and checkpoints")
    report["passed"] = True
    dump(output, report)
    print("PREFLIGHT", json.dumps(report["splits"]), flush=True)
    return report


class Progress:
    def __init__(self, run_dir, val_workers=0):
        self.run_dir = run_dir
        self.val_workers = val_workers
        self.started = time.time()
        self.last = 0
        self.batch = 0

    def setup_loaders(self, trainer):
        # Windows spawn loads a separate torch runtime per worker. Avoid eight
        # additional validation processes beside the four persistent train workers.
        loader = trainer.test_loader
        if loader.iterator is not None:
            loader.reset()
        loader.num_workers = self.val_workers
        if self.val_workers == 0:
            loader.prefetch_factor = None
        print(f"Validation loader workers: {loader.num_workers}", flush=True)

    def write(self, trainer, phase):
        total = len(trainer.train_loader)
        dump(self.run_dir / "status.json", {
            "phase": phase, "pid": os.getpid(), "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": min(trainer.epoch + 1, trainer.epochs), "epochs": trainer.epochs,
            "batch": self.batch, "batches_per_epoch": total,
            "elapsed_seconds": round(time.time() - self.started, 1),
            "metrics": {k: float(v) for k, v in trainer.metrics.items()},
            "best_fitness": trainer.best_fitness,
            "best_weights": str(trainer.best), "last_weights": str(trainer.last),
        })
        self.last = time.time()

    def epoch_start(self, trainer):
        self.batch = 0
        self.write(trainer, "training")

    def batch_end(self, trainer):
        self.batch += 1
        if time.time() - self.last >= 30:
            self.write(trainer, "training")

    def epoch_end(self, trainer):
        self.write(trainer, "epoch_finished")
        print("EPOCH_METRICS", trainer.epoch + 1, trainer.metrics, flush=True)


def evaluate(weights, data_path, run_dir, batch, workers):
    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont
    import random
    import yaml

    model = YOLO(str(weights), task="detect")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    expected = data["names"]
    expected = dict(enumerate(expected)) if isinstance(expected, list) else expected
    assert model.names == expected, "Checkpoint class names differ from dataset"
    result = model.val(data=str(data_path), split="test", imgsz=640, batch=batch,
                       workers=workers, device=0, cache=False, plots=True, verbose=False,
                       project=str(run_dir), name="test", exist_ok=True)
    report = {"split": "test", "weights": str(weights), "weights_sha256": sha256(weights),
              "metrics": {k: float(v) for k, v in result.results_dict.items()},
              "speed_ms_per_image": result.speed, "per_class": result.summary(),
              "scope": "Synthetic held-out test; not real-game screenshot performance"}
    dump(run_dir / "test_metrics.json", report)
    # Paired GT/prediction images with a legible Chinese-capable font.
    image_dir = Path(data["path"]) / data["test"]
    label_dir = image_dir.parent.parent / "labels" / "test"
    examples = random.Random(20261001).sample(sorted(image_dir.glob("*.png")), 20)
    previews = run_dir / "predictions"
    previews.mkdir(exist_ok=True)
    from ultralytics.utils import USER_CONFIG_DIR
    font = ImageFont.truetype(str(USER_CONFIG_DIR / "Arial.Unicode.ttf"), 13)
    outputs = model.predict(source=[str(p) for p in examples], imgsz=640, conf=0.25,
                            device=0, verbose=False, stream=True)
    manifest = []
    for path, prediction in zip(examples, outputs):
        original = Image.open(path).convert("RGB")
        w, h = original.size
        canvas = Image.new("RGB", (w * 2, h + 28), "#161b25")
        canvas.paste(original, (0, 28))
        canvas.paste(original, (w, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), "GT: bbox_visible", font=font, fill="white")
        draw.text((w + 8, 5), "YOLO prediction (conf >= 0.25)", font=font, fill="white")
        objects = []
        for line in (label_dir / (path.stem + ".txt")).read_text().splitlines():
            cls, x, y, bw, bh = map(float, line.split())
            objects.append((0, int(cls), None, [(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h]))
        for box in prediction.boxes:
            objects.append((w, int(box.cls.item()), float(box.conf.item()), box.xyxy[0].tolist()))
        for offset, cls, confidence, box in objects:
            x1, y1, x2, y2 = box
            color = tuple(60 + ((cls * p) % 180) for p in (47, 83, 113))
            draw.rectangle((x1+offset, y1+28, x2+offset, y2+28), outline=color, width=2)
            label = expected[cls] + (f" {confidence:.2f}" if confidence is not None else "")
            x, y = min(x1, w-120)+offset, max(28, y1+12)
            bbox = draw.textbbox((x, y), label, font=font)
            draw.rectangle(bbox, fill="black")
            draw.text((x, y), label, font=font, fill=color)
        target = previews / (path.stem + ".jpg")
        canvas.save(target, quality=92)
        manifest.append({"source": str(path), "preview": str(target), "detections": len(prediction.boxes)})
    dump(previews / "manifest.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "train_config.json"))
    parser.add_argument("--model", help="Use a specified local pretrained .pt")
    parser.add_argument("--resume", help="Resume interrupted training from last.pt")
    parser.add_argument("--evaluate", help="Evaluate an already trained best.pt without fitting")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    setup_ultralytics()
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(4)
    config = json.loads(resolved(args.config).read_text(encoding="utf-8"))
    val_workers = config.pop("val_workers", 0)
    data_path = resolved(config.pop("data"))
    model_path = resolved(args.model or config.pop("model"))
    config.pop("model", None)
    run_dir = ROOT / "runs" / config["name"]
    if args.resume:
        run_dir = resolved(args.resume).parent.parent
    if args.evaluate:
        run_dir = resolved(args.evaluate).parent.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "weights" / "last.pt").exists() and not (args.resume or args.evaluate or args.preflight_only):
        raise FileExistsError(f"Run already has weights. Use --resume, --evaluate, or a new config name: {run_dir}")
    started = time.time()
    try:
        dump(run_dir / "status.json", {"phase": "preflight", "pid": os.getpid(),
             "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "resume": args.resume})
        preflight(data_path, run_dir / "preflight.json")
        if args.preflight_only:
            return
        if not args.evaluate:
            dump(run_dir / "requested_config.json", {**config, "val_workers": val_workers, "model": str(model_path), "data": str(data_path)})
            model = YOLO(str(resolved(args.resume) if args.resume else model_path), task="detect")
            if model.task != "detect":
                raise ValueError(f"Expected detection checkpoint, got {model.task}")
            init_report = (f"resume_{time.strftime('%Y%m%d_%H%M%S')}.json" if args.resume else "initialization.json")
            dump(run_dir / init_report, {"checkpoint": str(model.ckpt_path),
                 "sha256": sha256(model.ckpt_path), "pretrained_classes": model.names,
                 "source": "resume_checkpoint" if args.resume else ("user_path" if args.model else "official_yolo11n_pretrained")})
            progress = Progress(run_dir, val_workers)
            model.add_callback("on_pretrain_routine_end", progress.setup_loaders)
            model.add_callback("on_train_epoch_start", progress.epoch_start)
            model.add_callback("on_train_batch_end", progress.batch_end)
            model.add_callback("on_fit_epoch_end", progress.epoch_end)
            if args.resume:
                model.train(resume=True)
            else:
                model.train(data=str(data_path), project=str(ROOT / "runs"), exist_ok=True, **config)
            weights = Path(model.trainer.best)
            del model
            torch.cuda.empty_cache()
        else:
            weights = resolved(args.evaluate)
        dump(run_dir / "status.json", {"phase": "testing", "pid": os.getpid(), "best_weights": str(weights)})
        report = evaluate(weights, data_path, run_dir, config["batch"], val_workers)
        dump(run_dir / "status.json", {"phase": "completed", "pid": os.getpid(),
             "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_seconds": time.time()-started,
             "best_weights": str(weights), "test_metrics": report["metrics"]})
        print("COMPLETED", report["metrics"], flush=True)
    except BaseException:
        dump(run_dir / "status.json", {"phase": "failed", "pid": os.getpid(), "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
