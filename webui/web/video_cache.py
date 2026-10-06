"""Finalize browser-playable H.264 videos before exposing them to the player."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import time

PLAYBACK_VERSION = 2


def ffmpeg_path():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    bundled = Path(__file__).resolve().parents[1] / 'tools/ffmpeg.exe'
    if bundled.is_file():
        return str(bundled)
    if importlib.util.find_spec('imageio_ffmpeg'):
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    raise ValueError('生成流畅视频预览需要 FFmpeg，请安装 FFmpeg 或 imageio-ffmpeg。')


def creationflags():
    return subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


class VideoEncoder:
    def __init__(self, folder, name, width, height, fps):
        self.folder, self.name = Path(folder), name
        self.target = self.folder / (name + '.mp4')
        self.temp = self.folder / (name + '.part.mp4')
        self.log = (self.folder / (name + '.ffmpeg.log')).open('wb')
        command = [ffmpeg_path(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                   '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{width}x{height}',
                   '-framerate', str(fps), '-i', 'pipe:0', '-an',
                   '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
                   '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
                   '-threads', '2', '-movflags', '+faststart', str(self.temp)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=self.log, creationflags=creationflags())
        except Exception:
            self.log.close()
            raise

    def write(self, bgr):
        try:
            self.process.stdin.write(bgr.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise ValueError('视频编码失败，请查看结果目录中的 ffmpeg.log。') from exc

    def finish(self):
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=120)
            if code or not self.temp.is_file() or self.temp.stat().st_size < 100:
                raise ValueError('视频编码失败，请查看结果目录中的 ffmpeg.log。')
            self.temp.replace(self.target)
        finally:
            if self.process.poll() is None:
                self.abort()
            self.log.close()

    def abort(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        self.log.close()
        self.temp.unlink(missing_ok=True)


def transcode(source, target, fps, duration=None, stopped=lambda: False):
    """Convert a legacy AVI or original media once, without loading any model."""
    target = Path(target)
    temp = target.with_name(target.stem + '.part.mp4')
    command = [ffmpeg_path(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', str(source)]
    if duration:
        command += ['-t', str(duration)]
    command += ['-an', '-vf', f'fps={fps},pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
                '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2',
                '-movflags', '+faststart', str(temp)]
    with target.with_suffix('.ffmpeg.log').open('wb') as log:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log, creationflags=creationflags())
        try:
            while process.poll() is None:
                if stopped():
                    raise InterruptedError('视频缓存转换已停止。')
                time.sleep(.1)
            if process.returncode or not temp.is_file() or temp.stat().st_size < 100:
                raise ValueError('历史视频转换失败，请查看结果目录中的 ffmpeg.log。')
            temp.replace(target)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            temp.unlink(missing_ok=True)
