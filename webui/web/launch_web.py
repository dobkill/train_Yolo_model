"""Start/reuse this local server, wait for model readiness, optionally open browser."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB.parent))
from runtime import dump, setup


def health(url):
    try:
        with urlopen(url + "/api/health", timeout=2) as response:
            value = json.load(response)
            if value.get("app") != "arad-local-detector-v1":
                raise RuntimeError("该端口被其他服务占用，请更换 -Port。")
            return value
    except (URLError, TimeoutError, ConnectionError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    setup()
    url = f"http://127.0.0.1:{args.port}"
    info = health(url)
    if info and (info.get('version') != 2 or info.get('build') != 'video-cache-2'):
        # Upgrade only a process that this launcher previously recorded and owns.
        import psutil
        record_path = WEB / 'service.json'
        record = json.loads(record_path.read_text(encoding='utf-8')) if record_path.is_file() else {}
        process = psutil.Process(record['pid']) if record.get('pid') else None
        expected = str(WEB / 'server.py').lower()
        command = process.cmdline() if process else []
        if expected not in [arg.lower() for arg in command] or str(args.port) not in command:
            raise RuntimeError('该端口仍有旧版服务，请使用 -Port 更换端口，或关闭旧服务后重新启动。')
        owned = process.children(recursive=True)
        for child in reversed(owned):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        _, alive = psutil.wait_procs([*owned, process], timeout=10)
        for child in alive:
            child.kill()
        info = None
    if not info:
        import psutil
        script = str(WEB / "server.py").lower()
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = process.info["cmdline"] or []
                if script in [part.lower() for part in cmd] and str(args.port) in cmd:
                    raise RuntimeError(f"服务 PID {process.pid} 正在初始化工作台，请稍后再次打开 {url}")
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
        log = WEB / "logs" / f"server_{datetime.now():%Y%m%d_%H%M%S}.log"
        log.parent.mkdir(exist_ok=True)
        command = [sys.executable, "-u", str(WEB / "server.py"), "--port", str(args.port)]
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=WEB, stdin=subprocess.DEVNULL,
                        stdout=handle, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        dump(WEB / "service.json", {"pid": process.pid, "url": url, "log": str(log),
             "script": str(WEB / "server.py"), "started": datetime.now().isoformat()})
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"服务启动失败，日志：{log}\n{log.read_text(encoding='utf-8')[-1500:]}")
            info = health(url)
            if info:
                break
            time.sleep(1)
        if not info:
            raise RuntimeError(f"模型仍在加载，日志：{log}；请稍后打开 {url}")
    print(f"模型测试工作台已就绪: {url} / SQLite 本地记录", flush=True)
    if not args.no_browser:
        webbrowser.open(url)


if __name__ == "__main__":
    main()
