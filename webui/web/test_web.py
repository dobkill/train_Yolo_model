"""Real API + Chromium integration tests. All generated data stays under web/qa."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(WEB.parent))
from library import Library

QA = WEB / 'qa'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def fixture():
    import cv2
    from PIL import Image
    import yaml
    folder = QA / 'fixtures'
    (folder / 'images').mkdir(parents=True, exist_ok=True)
    (folder / 'labels').mkdir(exist_ok=True)
    data = WEB.parents[1] / 'Datas/dataset_named_1000/yolo/detection'
    names = ['scene_022651', 'scene_022992', 'scene_023152', 'scene_023555']
    actual = []
    for name in names:
        source = data / f'images/test/{name}.png'
        if not source.is_file():
            source = next((data / 'images/test').glob('*.png'))
        shutil.copyfile(source, folder / 'images' / source.name)
        shutil.copyfile(data / 'labels/test' / (source.stem + '.txt'), folder / 'labels' / (source.stem + '.txt'))
        actual.append(folder / 'images' / source.name)
    # Dataset file has a deliberately stale root: adjacent images should resolve.
    config = yaml.safe_load((data.parents[1] / 'data_detection.yaml').read_text(encoding='utf-8'))
    config.update(path='Z:/moved/dataset', train='images', val='images', test='images')
    (folder / 'data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    import numpy as np
    frame = cv2.cvtColor(np.asarray(Image.open(actual[0]).convert('RGB')), cv2.COLOR_RGB2BGR)
    for name, count in [('短视频.avi', 8), ('停止测试.avi', 180)]:
        writer = cv2.VideoWriter(str(folder / name), cv2.VideoWriter_fourcc(*'MJPG'), 12, (frame.shape[1], frame.shape[0]))
        assert writer.isOpened(), 'Video fixture writer failed'
        for _ in range(count):
            writer.write(frame)
        writer.release()
    Image.new('RGBA', (60, 40), (255, 255, 255, 0)).save(folder / '透明图.png')
    return folder


def catalog_tests(folder):
    db = QA / 'unit-catalog/library.sqlite3'
    library = Library(db)
    first = library.register_dataset({'path': str(folder / 'data.yaml'), 'name': '移动后的数据集'})
    repeated = library.register_dataset({'path': str(folder / 'data.yaml'), 'name': '移动后的数据集'})
    assert first['id'] == repeated['id'] and first['image_count'] == 4
    disposable = QA / '失效记录.png'
    from PIL import Image
    Image.new('RGB', (4, 4)).save(disposable)
    entry = library.register_dataset({'path': str(disposable)})
    library.set_settings({'model_id': 'selection-test', 'dataset_id': first['id']})
    disposable.unlink()
    restarted = Library(db)
    assert not restarted.get('datasets', entry['id'])['valid']
    assert restarted.get_setting('dataset_id') == first['id']
    restarted.delete('datasets', entry['id'])
    assert (folder / 'data.yaml').is_file()
    return {'path_deduplication': True, 'startup_invalidation': True, 'selection_persistence': True, 'yaml_relocation': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default=str(WEB.parent / '.venv/Scripts/python.exe'))
    args = parser.parse_args()
    QA.mkdir(exist_ok=True)
    folder = fixture()
    checks = catalog_tests(folder)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    data = QA / 'integration-catalog'
    output = QA / 'integration-results'
    log = (QA / 'integration-server.log').open('w', encoding='utf-8')
    process = subprocess.Popen([args.python, str(WEB / 'server.py'), '--port', str(port),
                                '--data-dir', str(data), '--results-dir', str(output)],
                               stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)

    def request(path, body=None, headers=None, expected=200):
        req = Request(url + path, data=json.dumps(body).encode('utf-8') if body is not None else None,
                      headers={'Content-Type': 'application/json', **(headers or {})})
        try:
            with urlopen(req, timeout=120) as response:
                status, value = response.status, response.read()
        except HTTPError as exc:
            status, value = exc.code, exc.read()
        assert status == expected, (path, status, value[:500])
        return json.loads(value) if value[:1] in {b'{', b'['} else value

    def wait_job(job):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            latest = request('/api/jobs/' + job['id'])
            if latest['status'] not in {'running', 'queued', 'stopping'}:
                return latest
            time.sleep(.3)
        raise AssertionError('Test job timed out')

    try:
        for _ in range(100):
            try:
                info = request('/api/health')
                break
            except URLError:
                time.sleep(.1)
        else:
            raise AssertionError('Server did not start')
        assert info['ready'] and info['version'] == 2 and info['sqlite']
        weights = WEB.parents[1] / 'train/runs/named116_yolo11n/weights/best.pt'
        model = request('/api/models', {'path':str(weights),'task':'detect','name':'本地检测模型'})
        repeated = request('/api/models', {'path':str(weights),'task':'detect','name':'本地检测模型'})
        assert model['id'] == repeated['id']
        dataset = request('/api/datasets', {'path':str(folder),'name':'本地验证集'})
        files = request(f'/api/datasets/{dataset["id"]}/files?model_id={model["id"]}')
        assert files['total'] == 5
        image_file = next(f for f in files['items'] if f['name'].startswith('scene_022651'))
        assert len(request('/api/preview/' + image_file['id'])) > 1000
        params = {'conf':.35,'iou':.5,'imgsz':640,'device':'auto','stride':1}
        payload = {'model_id':model['id'],'dataset_id':dataset['id'],'kind':'image','parameters':params}
        job = request('/api/jobs', dict(payload, file_id=image_file['id']), expected=202)
        finished = wait_job(job)
        assert finished['status'] == 'completed', finished
        report = finished['latest_result']
        assert report['count'] > 0 and report['task'] == 'detect' and report['width'] == 640
        assert any(any('\u4e00' <= c <= '\u9fff' for c in d['class_name']) for d in report['detections'])
        assert request(report['original_url']) != request(report['annotated_url'])
        checks['real_detection'] = True
        print('PASS real detection', flush=True)
        request('/api/jobs', dict(payload, parameters={'conf':float('nan')}), expected=400)
        request('/api/models', {'path':str(weights),'task':'invalid'}, expected=400)
        request('/api/settings', {'task':'detect'}, headers={'Origin':'https://example.com'}, expected=403)
        request('/api/results/../../runtime.py', expected=404)
        checks['invalid_requests'] = True
        videos = request(f'/api/datasets/{dataset["id"]}/files?kind=video')['items']
        short = next(f for f in videos if f['name'] == '短视频.avi')
        raw = request('/api/media/' + short['id'], headers={'Range':'bytes=0-99'}, expected=206)
        assert len(raw) == 100
        video_job = request('/api/jobs', dict(payload,kind='video',file_id=short['id']), expected=202)
        video_done = wait_job(video_job)
        video = video_done['latest_result']
        assert video_done['status'] == 'completed' and video['processed_frames'] == 8, video_done
        assert video['playback_ready'] and video['video_url'].endswith('.mp4') and video['output_frames'] == 8
        import cv2
        decoded = cv2.VideoCapture(str(output/video['id']/'annotated.mp4'))
        assert int(decoded.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
        assert abs(decoded.get(cv2.CAP_PROP_FPS) - 12) < .01
        decoded.release()
        frame = request(f'/api/results/{video["id"]}/frame?index=3')
        assert frame['frame_index'] == 3 and frame['count'] > 0
        assert len(request(f'/api/results/{video["id"]}/frame.jpg?index=3')) > 1000
        assert len(request(video['video_url'])) > 1000
        checks['real_video_and_ranges'] = True
        print('PASS video/frame replay', flush=True)
        cached = wait_job(request('/api/jobs',dict(payload,kind='video',file_id=short['id']),expected=202))
        assert cached['cached'] == 1 and cached['latest_result']['id'] == video['id']
        sampled = wait_job(request('/api/jobs',dict(payload,kind='video',file_id=short['id'],parameters=dict(params,stride=3)),expected=202))
        assert sampled['latest_result']['processed_frames'] == 3 and sampled['latest_result']['output_frames'] == 8
        decoded = cv2.VideoCapture(str(output/sampled['latest_result']['id']/'annotated.mp4'))
        assert int(decoded.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
        assert abs(decoded.get(cv2.CAP_PROP_FPS) - 12) < .01
        decoded.release()
        checks['video_cache_reuse_and_full_frame_rate'] = True
        long = next(f for f in videos if f['name'] == '停止测试.avi')
        cancel = request('/api/jobs', dict(payload,kind='video',file_id=long['id']), expected=202)
        request('/api/jobs', dict(payload,kind='video',file_id=short['id']), expected=400)
        time.sleep(.5)
        request('/api/jobs/' + cancel['id'] + '/stop', {})
        assert wait_job(cancel)['status'] == 'cancelled'
        checks['stop_and_single_worker'] = True
        labeled = request('/api/datasets', {'path':str(folder/'data.yaml'),'name':'带标注验证集'})
        evaluation = request('/api/jobs', dict(payload,dataset_id=labeled['id'],mode='evaluate'), expected=202)
        evaluated = wait_job(evaluation)
        assert evaluated['status'] == 'completed' and evaluated['metrics']['metrics/mAP50(B)'] > 0, evaluated
        checks['real_evaluation'] = True
        print('PASS labeled evaluation', flush=True)
        # Save an untrained segmentation fixture to check the real segmentation path,
        # without downloading weights or changing the user's training files.
        seg_path = folder / 'pipeline-seg.pt'
        creation = subprocess.run([args.python, '-c',
            'from ultralytics import YOLO; import sys; YOLO("yolo11n-seg.yaml").save(sys.argv[1])', str(seg_path)],
            capture_output=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
        assert creation.returncode == 0, creation.stderr[-1000:]
        seg = request('/api/models', {'path':str(seg_path),'name':'分割流程验证模型（未训练）','task':'segment'})
        segmentation = request('/api/jobs', dict(payload,model_id=seg['id'],file_id=image_file['id'],parameters=dict(params,conf=0,imgsz=320)), expected=202)
        segmented = wait_job(segmentation)
        assert segmented['status'] == 'completed' and segmented['latest_result']['task'] == 'segment', segmented
        assert any(d['polygon'] for d in segmented['latest_result']['detections'])
        checks['real_segmentation_masks'] = True
        print('PASS segmentation/masks', flush=True)
        # ONNX is exported from a copy under QA; the training weights remain untouched.
        onnx_copy = folder / 'onnx-detection.pt'
        onnx_path = onnx_copy.with_suffix('.onnx')
        if not onnx_path.is_file():
            shutil.copyfile(weights, onnx_copy)
            exported = subprocess.run([args.python, '-c',
                'from ultralytics import YOLO; import sys,os; os.environ["YOLO_AUTOINSTALL"]="false"; YOLO(sys.argv[1]).export(format="onnx",imgsz=640,dynamic=True,simplify=False,opset=17,device="cpu")',str(onnx_copy)],
                capture_output=True,timeout=150,creationflags=subprocess.CREATE_NO_WINDOW)
            assert exported.returncode == 0, exported.stderr[-1500:]
        onnx_model=request('/api/models',{'path':str(onnx_path),'name':'ONNX 流程验证模型','task':'detect'})
        onnx_job=request('/api/jobs',dict(payload,model_id=onnx_model['id'],file_id=image_file['id']),expected=202)
        onnx_done=wait_job(onnx_job)
        assert onnx_done['status']=='completed' and onnx_done['latest_result']['count']>0,onnx_done
        assert onnx_done['latest_result']['device']=='cpu'
        checks['real_onnx_detection']=True
        print('PASS ONNX inference',flush=True)
        from playwright.sync_api import sync_playwright, expect
        errors = []
        request('/api/settings',{'model_id':model['id'],'dataset_id':dataset['id'],'task':'detect','kind':'image','parameters':params})
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            page=browser.new_page(viewport={'width':1720,'height':1000},accept_downloads=True)
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(url,wait_until='networkidle')
            expect(page.locator('#connection')).to_contain_text('已连接')
            expect(page.locator('.file-item')).to_have_count(5)
            expect(page.locator('#test-current')).to_be_enabled()
            expect(page.locator('#canvas')).to_be_visible()
            page.locator('#test-current').click()
            expect(page.locator('#notice')).to_contain_text('测试完成',timeout=90000)
            expect(page.locator('#target-count')).not_to_have_text('—')
            page.screenshot(path=str(QA/'workstation_desktop.png'),full_page=True)
            overlay=page.locator('#canvas').evaluate('c => c.toDataURL()')
            page.locator('[data-view="original"]').click()
            assert page.locator('#canvas').evaluate('c => c.toDataURL()') != overlay
            page.locator('[data-view="compare"]').click()
            assert page.locator('#canvas').evaluate('c => c.width') == 1280
            page.locator('[data-view="overlay"]').click()
            page.locator('#show-boxes').uncheck()
            page.locator('#show-names').uncheck()
            page.locator('#show-conf').uncheck()
            assert page.locator('#canvas').evaluate('c => c.toDataURL()') != overlay
            for id in ['show-boxes','show-names','show-conf']:
                page.locator('#'+id).check()
            with page.expect_download() as download:
                page.locator('#download-json').click()
            assert download.value.suggested_filename == 'result.json'
            page.locator('#register-model').click()
            page.locator('#entry-name').fill('本地检测模型')
            page.locator('#entry-path').fill(str(weights))
            page.locator('#save-entry').click()
            expect(page.locator('#register-dialog')).not_to_be_visible(timeout=15000)
            assert sum(m['path']==model['path'] for m in request('/api/health')['models']) == 1
            page.locator('[data-task="segment"]').click()
            expect(page.locator('#model-select')).to_have_value(seg['id'])
            expect(page.locator('#show-masks')).to_be_checked()
            page.locator('[data-task="detect"]').click()
            page.locator('[data-kind="video"]').click()
            expect(page.locator('.file-item')).to_have_count(2)
            page.locator('.file-item').filter(has_text='短视频.avi').click()
            expect(page.locator('#video-timeline')).to_be_visible()
            expect(page.locator('#video-player')).to_be_visible()
            deadline=time.monotonic()+20
            while page.locator('#video').evaluate('v=>v.readyState') < 2 and time.monotonic()<deadline:
                page.wait_for_timeout(100)
            assert page.locator('#video').evaluate('v=>v.readyState') >= 2
            playback_requests = []
            page.on('request', lambda req: playback_requests.append(req.url))
            page.locator('#play-frames').click()
            page.wait_for_timeout(450)
            assert page.locator('#video').evaluate('v=>v.currentTime') > .2
            assert not any('/frame.jpg' in path for path in playback_requests)
            assert page.locator('#video').evaluate('v=>v.getVideoPlaybackQuality().droppedVideoFrames') == 0
            page.locator('#play-frames').click()
            page.locator('#frame-seek').fill('3')
            expect(page.locator('#frame-label')).to_contain_text('第 4 帧',timeout=10000)
            page.screenshot(path=str(QA/'workstation_video.png'),full_page=True)
            page.locator('[data-view="compare"]').click()
            expect(page.locator('#video-reference')).to_be_visible()
            page.locator('[data-view="overlay"]').click()
            # Browsing image files must remain responsive during a long video task.
            page.locator('.file-item').filter(has_text='停止测试.avi').click()
            page.locator('#test-current').click()
            expect(page.locator('#stop')).to_be_enabled(timeout=10000)
            page.locator('[data-kind="image"]').click()
            expect(page.locator('.file-item')).to_have_count(5)
            expect(page.locator('#canvas')).to_be_visible(timeout=10000)
            expect(page.locator('#processing')).not_to_be_visible()
            page.locator('#stop').click()
            expect(page.locator('#stop')).to_be_disabled(timeout=30000)
            expect(page.locator('#notice')).to_contain_text('已停止',timeout=30000)
            checks['switch_images_during_video_task'] = True
            checks['native_video_playback_without_frame_images'] = True
            page.locator('#history-library').click()
            expect(page.locator('.history-row')).not_to_have_count(0)
            page.locator('#history-dialog .close-dialog').click()
            page.locator('[data-kind="image"]').click()
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(QA/'workstation_mobile.png'),full_page=True)
            page.set_viewport_size({'width':1280,'height':900})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(QA/'workstation_1280.png'),full_page=True)
            assert not errors, errors
            browser.close()
        checks['browser_ui_interactions'] = True
        checks['responsive_layouts'] = True
        checks['browser_errors'] = errors
        report = {'passed':True,'checks':checks,'checked_at':time.strftime('%Y-%m-%d %H:%M:%S'),
                  'segmentation_fixture':'Untrained model used only to validate segmentation execution and mask rendering.'}
        (QA/'test_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    finally:
        # Windows venv python.exe is a launcher; terminate its server child too.
        import psutil
        try:
            parent=psutil.Process(process.pid)
            owned=parent.children(recursive=True)
            for child in reversed(owned):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            parent.terminate()
            _,alive=psutil.wait_procs([*owned,parent],timeout=10)
            for child in alive:
                child.kill()
        except psutil.NoSuchProcess:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


if __name__ == '__main__':
    main()
