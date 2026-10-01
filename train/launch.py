"""Launch a detached local trainer with UTF-8 logs and an inspectable PID."""
import datetime
import os
import subprocess
import sys
from runtime import ROOT, dump, setup

setup()
if __name__ == "__main__":
    import psutil
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info["cmdline"] or []
            if any(str(ROOT / "train.py").lower() == p.lower() for p in cmd):
                raise SystemExit(f"This trainer is already running, PID {proc.pid}")
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = ROOT / "logs" / f"train_{stamp}.log"
    command = [sys.executable, "-u", str(ROOT / "train.py"), *sys.argv[1:]]
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    dump(ROOT / "launch.json", {"pid": proc.pid, "log": str(log), "command": command, "started": stamp})
    print(f"Started trainer PID {proc.pid}\nLog: {log}")
