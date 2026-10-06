"""Local HTTP API for the model testing workstation."""
import argparse
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB.parent))
from runtime import setup
from library import Library
from engine import Engine
from jobs import Jobs

APP_ID = 'arad-local-detector-v1'


class Handler(BaseHTTPRequestHandler):
    server_version = 'LocalModelWorkbench/2.0'

    def send_headers(self, size, mime, status=200, extra=None):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', (extra or {}).get('Cache-Control', 'no-store'))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        for key, value in (extra or {}).items():
            if key != 'Cache-Control':
                self.send_header(key, value)
        self.end_headers()

    def reply(self, value, status=200, mime='application/json; charset=utf-8'):
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_headers(len(value), mime, status)
        try:
            self.wfile.write(value)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        host = urlsplit('http://' + self.headers.get('Host', ''))
        if host.hostname not in {'127.0.0.1', 'localhost'} or host.port != self.server.server_port:
            raise PermissionError('服务仅接受本机访问。')
        origin = self.headers.get('Origin')
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.port != self.server.server_port:
                raise PermissionError('请从本地工作台提交请求。')
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise PermissionError('不接受外部网页的请求。')

    def file(self, path, download=False):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError('文件已移动或删除。')
        size = path.stat().st_size
        start, end, status = 0, max(0, size - 1), 200
        extra = {'Accept-Ranges': 'bytes'}
        if path.suffix.lower() in {'.mp4', '.jpg'} and path.parent.name != 'static':
            extra['Cache-Control'] = 'private, max-age=3600'
        requested = self.headers.get('Range')
        if requested and size:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                return self.reply({'error': '无效的媒体范围。'}, 416)
            if match[1]:
                start = int(match[1])
                end = min(end, int(match[2])) if match[2] else end
            else:
                start = max(0, size - int(match[2]))
            if start > end or start >= size:
                self.send_headers(0, 'application/octet-stream', 416, {'Content-Range': f'bytes */{size}'})
                return
            status = 206
            extra['Content-Range'] = f'bytes {start}-{end}/{size}'
        mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if path.suffix.lower() in {'.html', '.css', '.js', '.json', '.svg'}:
            mime += '; charset=utf-8'
        if download:
            extra['Content-Disposition'] = f'attachment; filename="{path.name}"'
        remaining = end - start + 1 if size else 0
        self.send_headers(remaining, mime, status, extra)
        try:
            with path.open('rb') as stream:
                stream.seek(start)
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def thumbnail(self, file):
        from PIL import Image, ImageOps
        cache = self.server.data / 'thumbnails'
        cache.mkdir(exist_ok=True)
        path = Path(file['path'])
        if not path.is_file():
            raise FileNotFoundError('源文件已移动或删除。')
        target = cache / f'{file["id"]}_{path.stat().st_mtime_ns}.jpg'
        if not target.exists():
            if file['kind'] == 'video':
                import cv2
                capture = cv2.VideoCapture(str(path))
                try:
                    ok, bgr = capture.read()
                    if not ok:
                        raise ValueError('无法解码视频缩略图。')
                    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                finally:
                    capture.release()
            else:
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert('RGB')
            image.thumbnail((160, 100))
            stream = io.BytesIO()
            image.save(stream, 'JPEG', quality=80)
            target.write_bytes(stream.getvalue())
        return self.file(target)

    def do_GET(self):
        try:
            self.local_request()
            url = urlsplit(self.path)
            path, query = unquote(url.path), parse_qs(url.query)
            library = self.server.library
            if path == '/api/health':
                return self.reply({'app': APP_ID, 'version': 2, 'build': 'video-cache-2', 'ready': True, 'device': '按测试参数选择',
                                   'classes': None, 'sqlite': True, 'database': str(library.db),
                                   'models': library.rows('models'), 'datasets': library.rows('datasets'),
                                   'settings': library.settings(),
                                   'active_job': self.server.jobs.get(self.server.jobs.active['id']) if self.server.jobs.active else None,
                                   'invalid': [r for table in ('models', 'datasets') for r in library.rows(table) if not r['valid']]})
            if path == '/api/history':
                return self.reply(library.history())
            match = re.fullmatch(r'/api/datasets/([0-9a-f]{32})/files', path)
            if match:
                return self.reply(library.files(match[1], query.get('kind', ['image'])[0],
                                               query.get('search', [''])[0][:200],
                                               max(0, int(query.get('offset', ['0'])[0])),
                                               min(200, max(1, int(query.get('limit', ['80'])[0]))),
                                               query.get('model_id', [''])[0]))
            match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})', path)
            if match:
                return self.reply(self.server.jobs.get(match[1]))
            match = re.fullmatch(r'/api/(media|thumb|preview)/([0-9a-f]{32})', path)
            if match:
                file = library.get('files', match[2])
                if not library.get('datasets', file['dataset_id'])['valid']:
                    raise ValueError('数据集已失效，请重新检查。')
                if match[1] == 'preview':
                    from PIL import Image
                    if file['kind'] == 'image':
                        image = self.server.engine.read_image(file['path'])
                    else:
                        import cv2
                        capture = cv2.VideoCapture(file['path'])
                        try:
                            ok, bgr = capture.read()
                            if not ok:
                                raise ValueError('无法读取视频画面。')
                            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        finally:
                            capture.release()
                    stream = io.BytesIO()
                    image.save(stream, 'JPEG', quality=95)
                    return self.reply(stream.getvalue(), mime='image/jpeg')
                return self.thumbnail(file) if match[1] == 'thumb' else self.file(file['path'])
            match = re.fullmatch(r'/api/results/([0-9a-f]{32})/(original\.png|annotated\.png|original\.mp4|annotated\.mp4|result\.json|annotated\.avi|metrics\.json|frame|frame\.jpg|playback)', path)
            if match:
                folder = self.server.engine.output / match[1]
                if not folder.is_dir():
                    raise FileNotFoundError('测试结果不存在。')
                if match[2] == 'playback':
                    return self.reply(self.server.engine.playback_status(match[1]))
                if match[2] in {'frame', 'frame.jpg'}:
                    if not (folder / 'frames.sqlite3').is_file():
                        raise ValueError('该结果没有视频帧记录。')
                    frame = max(0, int(query.get('index', ['0'])[0]))
                    if match[2] == 'frame':
                        return self.reply(self.server.engine.frame(match[1], frame))
                    return self.reply(self.server.engine.frame_image(match[1], frame), mime='image/jpeg')
                return self.file(folder / match[2], 'download' in query)
            if path in {'/', '/index.html', '/app.css', '/app.js', '/favicon.svg'}:
                return self.file(WEB / 'static' / ('index.html' if path == '/' else path[1:]))
            raise FileNotFoundError('页面或接口不存在。')
        except Exception as exc:
            self.error(exc)

    def body(self):
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= 65536:
            self.close_connection = True
            raise ValueError('请求数据为空或超过 64 KB。')
        self.connection.settimeout(30)
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError('请求必须为 JSON 对象。')
        return value

    def do_POST(self):
        try:
            self.local_request()
            body = self.body()
            path = urlsplit(self.path).path
            library = self.server.library
            active = self.server.jobs.active
            if path in {'/api/remove', '/api/validate', '/api/models', '/api/datasets'} and active and active['status'] in {'queued', 'running', 'stopping'}:
                raise ValueError('测试期间请等待结束或停止后再修改登记记录。')
            if path == '/api/models':
                return self.reply(library.register_model(body))
            if path == '/api/datasets':
                return self.reply(library.register_dataset(body))
            if path == '/api/settings':
                allowed = {'model_id', 'dataset_id', 'task', 'kind', 'parameters', 'mode'}
                library.set_settings({k: v for k, v in body.items() if k in allowed})
                return self.reply({'saved': True})
            if path == '/api/validate':
                return self.reply({'invalid': library.validate()})
            if path == '/api/remove':
                library.delete(body.get('category'), str(body.get('id', '')))
                return self.reply({'removed': True})
            if path == '/api/jobs':
                return self.reply(self.server.jobs.start(body), 202)
            match = re.fullmatch(r'/api/results/([0-9a-f]{32})/playback', path)
            if match:
                return self.reply(self.server.engine.prepare_playback(match[1]))
            match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})/stop', path)
            if match:
                return self.reply(self.server.jobs.cancel(match[1]))
            if path == '/api/picker':
                kind = body.get('kind')
                if kind not in {'model', 'folder', 'yaml', 'media'}:
                    raise ValueError('文件选择类型无效。')
                if not self.server.picker_lock.acquire(blocking=False):
                    raise ValueError('已有文件选择窗口打开，请先完成选择。')
                try:
                    result = subprocess.run([sys.executable, str(WEB / 'pick_local.py'), kind],
                                            capture_output=True, text=True, encoding='utf-8', timeout=300,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    if result.returncode:
                        raise ValueError('无法打开系统选择窗口，请直接填写本地完整路径。')
                    return self.reply(json.loads(result.stdout))
                finally:
                    self.server.picker_lock.release()
            raise FileNotFoundError('接口不存在。')
        except Exception as exc:
            self.error(exc)

    def error(self, exc):
        if isinstance(exc, PermissionError):
            status = 403
        elif isinstance(exc, FileNotFoundError):
            status = 404
        elif isinstance(exc, (ValueError, KeyError, TypeError, TimeoutError, OSError)):
            status = 400
        else:
            traceback.print_exc()
            status = 500
        self.reply({'error': str(exc) if status != 500 else '操作失败，详细原因已记录在本地日志。'}, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--data-dir', type=Path, default=WEB / 'data')
    parser.add_argument('--results-dir', type=Path, default=WEB / 'results')
    args = parser.parse_args()
    setup()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    server.data = args.data_dir.resolve()
    try:
        print('Checking SQLite model/dataset catalog...', flush=True)
        server.library = Library(server.data / 'library.sqlite3')
        server.engine = Engine(server.library, args.results_dir.resolve())
        server.jobs = Jobs(server.library, server.engine)
        server.picker_lock = threading.Lock()
        print(f'READY http://127.0.0.1:{args.port} / SQLite / {len(server.library.invalid)} invalid records', flush=True)
        server.serve_forever(poll_interval=.5)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
