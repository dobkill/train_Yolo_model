"""Print the latest detached training job's status and whether its PID is alive."""
import json
from runtime import ROOT

if __name__ == "__main__":
    import psutil
    launch_path = ROOT / "launch.json"
    if not launch_path.exists():
        raise SystemExit("No job launched yet")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    running = False
    try:
        process = psutil.Process(launch["pid"])
        running = any(str(ROOT / "train.py").lower() == arg.lower() for arg in process.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    print(json.dumps({"process_running": running, **launch}, ensure_ascii=False, indent=2))
    for path in sorted((ROOT / "runs").glob("*/status.json")):
        print(path)
        print(path.read_text(encoding="utf-8"))
