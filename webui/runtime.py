"""Web UI runtime. Keeps web caches in Work/webui; model weights stay in Work/train."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

WEBUI = Path(__file__).resolve().parent      # Work/webui
WORK = WEBUI.parent                          # Work
TRAIN = WORK / "train"                       # trained weights / metrics live here

# Backwards-compatible alias: registry paths below ROOT point into Work/train.
ROOT = TRAIN


def setup():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.update({
        "YOLO_CONFIG_DIR": str(WEBUI / ".ultralytics"),
        "MPLCONFIGDIR": str(WEBUI / ".matplotlib"),
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    })
    for folder in (".ultralytics", ".matplotlib"):
        (WEBUI / folder).mkdir(exist_ok=True)
    os.chdir(WEBUI)


def setup_ultralytics():
    from ultralytics import settings
    from ultralytics.utils import USER_CONFIG_DIR
    settings.update({
        "datasets_dir": str(WORK), "weights_dir": str(TRAIN / "pretrained"),
        "runs_dir": str(TRAIN / "runs"), "sync": False,
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
