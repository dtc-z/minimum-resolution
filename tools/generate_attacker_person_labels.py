"""Create person boxes for the 240-frame research sample.

Boxes come from the bundled COCO-pretrained YOLO26n model. The requested
semantic role is copied from the user specification and is not inferred from
pixels. This output is machine-generated, not human ground truth.
"""
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "final" / "automated_audit" / "sample.csv"
OUTPUT = ROOT / "final" / "annotations" / "ai_attacker_person_boxes_240.csv"
MODEL_PATH = ROOT / "yolo26n.pt"


def main():
    sample = pd.read_csv(SAMPLE)
    model = YOLO(str(MODEL_PATH))
    person_id = next(i for i, name in model.names.items() if name == "person")
    rows = []
    records = sample.to_dict("records")
    for start in range(0, len(records), 4):
        batch = records[start : start + 4]
        results = model.predict(
            source=[str(item["image_path"]) for item in batch],
            classes=[person_id],
            conf=0.1,
            imgsz=800,
            batch=4,
            verbose=False,
        )
        for item, result in zip(batch, results):
            candidates = []
            for box in result.boxes:
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
                candidates.append((area, conf, xyxy))
            candidates.sort(reverse=True)
            chosen = candidates[0] if candidates else (0.0, 0.0, [None] * 4)
            x1, y1, x2, y2 = chosen[2]
            rows.append({
                "dataset": item["dataset"],
                "frame_number": int(item["frame_number"]),
                "image_width": 800,
                "image_height": 600,
                "class_label": "attacker",
                "class_label_source": "user_requested_not_image_verified",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "detector": "YOLO26n_COCO_person",
                "detector_confidence": chosen[1],
                "person_candidates": len(candidates),
                "label_source": "machine_generated_unverified",
                "selection_ambiguous_multiple_people": len(candidates) > 1,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)
    print(f"rows={len(out)} detections={int(out['x1'].notna().sum())} "
          f"multi_person_frames={int(out['selection_ambiguous_multiple_people'].sum())} "
          f"median_confidence={out['detector_confidence'].median():.4f}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
