"""Run the trained detector on an image, directory, or video."""
import argparse
from datetime import datetime
from pathlib import Path
from runtime import ROOT, dump, resolved, setup, setup_ultralytics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Image, image directory, or video path")
    parser.add_argument("--weights", default="runs/named116_yolo11n/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    setup()
    setup_ultralytics()
    from ultralytics import YOLO
    weights = resolved(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"No trained checkpoint yet: {weights}")
    if not source.exists():
        raise FileNotFoundError(source)
    name = datetime.now().strftime("predict_%Y%m%d_%H%M%S")
    output = ROOT / "inference" / name
    model = YOLO(str(weights), task="detect")
    predictions = model.predict(source=str(source), imgsz=640, conf=args.conf,
                                device=args.device, save=True, save_txt=True,
                                save_conf=True, project=str(output.parent),
                                name=name, exist_ok=True, stream=True)
    # Stream rows to avoid retaining a long video's predictions in RAM.
    output.mkdir(parents=True, exist_ok=True)
    with (output / "detections.jsonl").open("w", encoding="utf-8") as handle:
        import json
        for index, result in enumerate(predictions):
            row = {"index": index, "source": result.path, "instances": []}
            for box in result.boxes:
                cls = int(box.cls.item())
                row["instances"].append({"class_id": cls, "class_name": result.names[cls],
                      "confidence": float(box.conf.item()), "bbox_xyxy": box.xyxy[0].tolist()})
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    dump(output / "inference.json", {"weights": str(weights), "source": str(source), "confidence": args.conf})
    print(output)


if __name__ == "__main__":
    main()
