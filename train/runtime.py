"""Keep this project's runtime files in Work/train, including on Windows spawn."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent


def setup():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.update({
        "YOLO_CONFIG_DIR": str(ROOT / ".ultralytics"),
        "MPLCONFIGDIR": str(ROOT / ".matplotlib"),
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    })
    for folder in (".ultralytics", ".matplotlib", "pretrained", "runs", "logs"):
        (ROOT / folder).mkdir(exist_ok=True)
    os.chdir(ROOT)


def setup_ultralytics():
    from ultralytics import settings
    from ultralytics.utils import USER_CONFIG_DIR
    settings.update({
        "datasets_dir": str(ROOT.parent), "weights_dir": str(ROOT / "pretrained"),
        "runs_dir": str(ROOT / "runs"), "sync": False,
        **{key: False for key in ("clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb")},
    })
    # Use installed fonts for Chinese labels, avoiding font downloads.
    for filename, source in (("Arial.Unicode.ttf", "simhei.ttf"), ("Arial.ttf", "arial.ttf")):
        target = USER_CONFIG_DIR / filename
        source = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / source
        if source.is_file() and not target.exists():
            shutil.copyfile(source, target)
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Arial", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def resolved(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
