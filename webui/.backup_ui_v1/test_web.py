"""Exercise actual local model inference, browser upload, previews and downloads."""
import io
import json
from pathlib import Path
import time
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright, expect

WEB = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
QA = WEB / "qa"
URL = "http://127.0.0.1:8766"


def request(path, payload=None, headers=None):
    req = Request(URL + path, data=payload, headers=headers or {})
    try:
        with urlopen(req, timeout=60) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def main():
    QA.mkdir(exist_ok=True)
    status, body = request("/api/health")
    info = json.loads(body)
    assert status == 200 and info["ready"] and info["classes"] == 116
    source = request(info["examples"][0]["url"])[1]
    upload = QA / "上传示例.png"
    upload.write_bytes(source)
    Image.new("RGB", (640, 480), "white").save(QA / "blank.png")
    cases = {
        "invalid_image": request("/api/predict", b"this is not an image")[0],
        "invalid_confidence": request("/api/predict?conf=nan", source)[0],
        "invalid_size": request("/api/predict?imgsz=777", source)[0],
        "foreign_origin": request("/api/predict", source, {"Origin": "https://example.com"})[0],
        "path_traversal": request("/api/results/../../train_config.json")[0],
    }
    assert cases == {"invalid_image": 400, "invalid_confidence": 400, "invalid_size": 400,
                     "foreign_origin": 403, "path_traversal": 404}, cases
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",
                                   headless=True, args=["--disable-extensions", "--no-first-run"])
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1,
                                      accept_downloads=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="networkidle")
        expect(page.locator("#model-state")).to_have_text("模型已就绪")
        expect(page.locator(".example-button")).to_have_count(4)
        expect(page.locator("#detect")).to_be_disabled()
        page.screenshot(path=str(QA / "desktop_empty.png"), full_page=True)
        page.locator("#file-input").set_input_files([str(upload), str(QA / "blank.png")])
        expect(page.locator(".queue-item")).to_have_count(2)
        expect(page.locator("#detect")).to_be_enabled()
        with page.expect_response(lambda response: "/api/predict?" in response.url, timeout=60000) as response:
            page.locator("#detect").click()
        assert response.value.status == 200
        expect(page.locator("#notice")).to_contain_text("识别完成", timeout=60000)
        result = json.loads(request(page.locator("#download-json").get_attribute("href"))[1])
        assert result["count"] > 0 and result["weights_sha256"] == info["weights_sha256"]
        assert result["filename"] == "上传示例.png"
        assert result["width"] == 640 and result["height"] == 480
        assert any(any("\u4e00" <= ch <= "\u9fff" for ch in d["class_name"]) for d in result["detections"])
        for d in result["detections"]:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
            assert 0 <= d["class_id"] < 116 and d["confidence"] >= .25
        expect(page.locator("#target-count")).to_have_text(str(result["count"]))
        expect(page.locator(".result-row")).to_have_count(result["count"])
        page.screenshot(path=str(QA / "desktop_result.png"), full_page=True)
        overlay = page.locator("#canvas").evaluate("canvas => canvas.toDataURL()")
        page.locator("#view-original").click()
        original = page.locator("#canvas").evaluate("canvas => canvas.toDataURL()")
        assert overlay != original, "Canvas must actually draw boxes"
        page.locator("#view-boxes").click()
        first_class = result["detections"][0]["class_id"]
        page.locator("#class-filter").select_option(str(first_class))
        expect(page.locator(".result-row")).to_have_count(sum(d["class_id"] == first_class for d in result["detections"]))
        page.locator("#class-filter").select_option("all")
        with page.expect_download() as downloaded:
            page.locator("#download-image").click()
        downloaded.value.save_as(str(QA / "download_annotated.png"))
        with Image.open(io.BytesIO(request(result["original_url"])[1])) as normalized:
            with Image.open(QA / "download_annotated.png") as annotated:
                assert annotated.size == normalized.size
                assert ImageChops.difference(normalized.convert("RGB"), annotated.convert("RGB")).getbbox()
        with page.expect_download() as downloaded_json:
            page.locator("#download-json").click()
        downloaded_json.value.save_as(str(QA / "download_result.json"))
        assert json.loads((QA / "download_result.json").read_text(encoding="utf-8"))["id"] == result["id"]
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Mobile horizontal overflow"
        page.screenshot(path=str(QA / "mobile_result.png"), full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.locator(".queue-item").nth(1).click()
        expect(page.locator("#target-count")).to_have_text("—")
        expect(page.locator("#download-image")).to_have_attribute("aria-disabled", "true")
        page.locator("#confidence").focus()
        page.locator("#confidence").press("End")
        with page.expect_response(lambda response: "/api/predict?" in response.url, timeout=60000) as empty_response:
            page.locator("#detect").click()
        expect(page.locator("#notice")).to_contain_text("识别完成", timeout=60000)
        empty = json.loads(request(page.locator("#download-json").get_attribute("href"))[1])
        assert empty_response.value.status == 200 and empty["count"] == 0, empty
        expect(page.locator("#result-list")).to_contain_text("未检测到目标")
        page.screenshot(path=str(QA / "empty_detection.png"), full_page=True)
        page.locator(".queue-item").nth(0).click()
        expect(page.locator("#target-count")).to_have_text(str(result["count"]))
        page.locator(".example-button").nth(1).click()
        expect(page.locator(".queue-item")).to_have_count(3)
        # Browser drag and drop follows the same real file-upload path.
        data = page.evaluate_handle("""() => {
            const dt = new DataTransfer();
            const canvas = document.createElement('canvas'); canvas.width=40;canvas.height=40;
            const bytes=Uint8Array.from(atob(canvas.toDataURL('image/png').split(',')[1]), c=>c.charCodeAt(0));
            dt.items.add(new File([bytes], 'drop.png', {type:'image/png'}));return dt;
        }""")
        page.locator("#dropzone").dispatch_event("drop", {"dataTransfer": data})
        expect(page.locator(".queue-item")).to_have_count(4)
        assert not errors, errors
        browser.close()
    report = {"passed": True, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "api_error_cases": cases, "browser_errors": errors,
              "positive_detections": result["count"], "blank_detections": empty["count"],
              "model_sha256": info["weights_sha256"], "example_result_id": result["id"],
              "checks": ["actual_best_checkpoint_inference", "Chinese_filename_and_labels", "multiple_files",
                         "bbox_coordinates", "canvas_boxes", "original_toggle", "class_filter", "PNG_download",
                         "JSON_download", "mobile_layout", "no_detection_state", "switch_result_state", "demo", "drag_drop"]}
    (QA / "test_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
