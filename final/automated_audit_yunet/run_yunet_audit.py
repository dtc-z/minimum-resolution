"""Run an automatic YuNet face-box comparison on the fixed 240-frame audit sample."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SAMPLE = Path(__file__).resolve().parents[1] / "automated_audit" / "sample.csv"
WEIGHTS = ROOT / "model" / "face_detection_yunet_2023mar.onnx"
OUT = ROOT / "model_output"
OVERLAYS = OUT / "overlays"
INPUT_SIZE = (640, 640)
SCORE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.30
TOP_K = 5000
MODEL_NAME = "OpenCV Zoo YuNet face_detection_yunet_2023mar"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def make_canvas(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(INPUT_SIZE[0] / width, INPUT_SIZE[1] / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    offset_x = (INPUT_SIZE[0] - resized_width) // 2
    offset_y = (INPUT_SIZE[1] - resized_height) // 2
    canvas = np.zeros((INPUT_SIZE[1], INPUT_SIZE[0], 3), dtype=np.uint8)
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    return canvas, scale, offset_x, offset_y


def draw_overlay(image, boxes, chosen, chosen_score, reference, path: Path) -> None:
    canvas = image.copy()
    for box, score in boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 255), 1)
        cv2.putText(canvas, f"YuNet {score:.2f}", (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 180), 1, cv2.LINE_AA)
    x1, y1, x2, y2 = [int(round(value)) for value in reference]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(canvas, "USER-DRAWN REFERENCE", (x1, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1, cv2.LINE_AA)
    if chosen is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in chosen]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(canvas, f"YuNet largest {chosen_score:.2f}", (x1, min(canvas.shape[0] - 8, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 0), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def main() -> None:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    sample = pd.read_csv(SAMPLE)
    if len(sample) != 240:
        raise ValueError(f"Expected 240 fixed audit frames, found {len(sample)}")
    if not sample.box_source.eq("user_drawn_manual_annotation").all():
        raise ValueError("Reference boxes must be tagged as user-drawn annotations")

    detector = cv2.FaceDetectorYN.create(
        str(WEIGHTS), "", INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
    )
    rows = []
    for index, item in sample.iterrows():
        image = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        row = item.to_dict()
        row.update({
            "automated_detector": MODEL_NAME,
            "person_role_label": "attacker",
            "person_role_source": "user_assumption",
            "score_threshold": SCORE_THRESHOLD,
            "nms_threshold": NMS_THRESHOLD,
            "image_read_status": "ok" if image is not None else "unreadable",
            "face_count": 0,
            "auto_target_x1": np.nan,
            "auto_target_y1": np.nan,
            "auto_target_x2": np.nan,
            "auto_target_y2": np.nan,
            "auto_target_score": np.nan,
            "iou_auto_vs_manual_box": np.nan,
            "auto_center_distance_px": np.nan,
            "auto_to_manual_area_ratio": np.nan,
            "all_detections_json": "[]",
            "error": "",
        })
        if image is None:
            row["error"] = "cv2.imread returned None"
            rows.append(row)
            continue

        try:
            canvas, scale, offset_x, offset_y = make_canvas(image)
            _, faces = detector.detect(canvas)
            height, width = image.shape[:2]
            detections = []
            if faces is not None:
                for face in faces:
                    x, y, box_width, box_height = map(float, face[:4])
                    box = [
                        float(np.clip((x - offset_x) / scale, 0, width - 1)),
                        float(np.clip((y - offset_y) / scale, 0, height - 1)),
                        float(np.clip((x + box_width - offset_x) / scale, 0, width - 1)),
                        float(np.clip((y + box_height - offset_y) / scale, 0, height - 1)),
                    ]
                    detections.append({"box": box, "score": float(face[-1])})
            row["face_count"] = len(detections)
            row["all_detections_json"] = json.dumps(detections, separators=(",", ":"))
            chosen = max(
                detections,
                key=lambda detection: max(0.0, detection["box"][2] - detection["box"][0])
                * max(0.0, detection["box"][3] - detection["box"][1]),
                default=None,
            )
            if chosen:
                box = chosen["box"]
                reference = [float(item[key]) for key in ("x1", "y1", "x2", "y2")]
                row.update({
                    "auto_target_x1": box[0], "auto_target_y1": box[1],
                    "auto_target_x2": box[2], "auto_target_y2": box[3],
                    "auto_target_score": chosen["score"],
                    "iou_auto_vs_manual_box": iou(box, reference),
                    "auto_center_distance_px": float(np.hypot(
                        (box[0] + box[2] - reference[0] - reference[2]) / 2,
                        (box[1] + box[3] - reference[1] - reference[3]) / 2,
                    )),
                    "auto_to_manual_area_ratio": (
                        max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                        / max(1.0, (reference[2] - reference[0]) * (reference[3] - reference[1]))
                    ),
                })
            target = OVERLAYS / str(item.dataset) / f"{int(item.frame_number):08d}.jpg"
            draw_overlay(
                image, [(d["box"], d["score"]) for d in detections],
                chosen["box"] if chosen else None, chosen["score"] if chosen else None,
                [float(item.x1), float(item.y1), float(item.x2), float(item.y2)], target,
            )
        except Exception as exc:
            row["image_read_status"] = "inference_error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if (index + 1) % 10 == 0:
            print(f"processed {index + 1}/240", flush=True)

    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "model_proposals.csv", index=False, encoding="utf-8-sig")
    successful = result[result.auto_target_score.notna()]
    metadata = {
        "status": "automated_yunet_proposals_compared_with_user_drawn_manual_boxes",
        "automated_proposals": True,
        "human_annotation": False,
        "reference_box_source": "user_drawn_manual_annotation",
        "person_role_assumption": "attacker",
        "person_role_source": "user_assumption; not classified by the detector",
        "paper_modified": False,
        "occlusion": "assumed_none based on user-provided statement; not independently assessed",
        "sample_count": len(sample),
        "sample_seed": 391,
        "sample_rule": "20 per test sequence x eye-distance size stratum; 4 sequences x 3 strata",
        "target_box_rule": "largest detected face box per frame; all detector boxes retained in CSV",
        "model": MODEL_NAME,
        "model_source": "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
        "weights": str(WEIGHTS),
        "weights_sha256": sha256(WEIGHTS),
        "license": "OpenCV Zoo model files: MIT License",
        "opencv_version": cv2.__version__,
        "detector_input_size": INPUT_SIZE,
        "preprocessing": "aspect-preserving resize and centered black padding to 640x640; boxes mapped back to source coordinates",
        "score_threshold": SCORE_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "inference_device": "CPU via OpenCV DNN",
        "images_with_detection": len(successful),
        "images_without_detection": int(result.auto_target_score.isna().sum()),
        "mean_model_confidence": float(successful.auto_target_score.mean()) if len(successful) else None,
        "median_iou_with_manual_boxes": float(successful.iou_auto_vs_manual_box.median()) if len(successful) else None,
        "median_center_distance_px": float(successful.auto_center_distance_px.median()) if len(successful) else None,
        "median_auto_to_manual_area_ratio": float(successful.auto_to_manual_area_ratio.median()) if len(successful) else None,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result.assign(detected=result.auto_target_score.notna()).groupby(
        ["dataset", "size_stratum"], dropna=False
    ).agg(
        samples=("image_path", "count"),
        detections=("detected", "sum"),
        median_iou=("iou_auto_vs_manual_box", "median"),
        median_center_distance_px=("auto_center_distance_px", "median"),
        median_auto_to_manual_area_ratio=("auto_to_manual_area_ratio", "median"),
        median_confidence=("auto_target_score", "median"),
    ).reset_index().to_csv(OUT / "summary_by_sequence_and_size.csv", index=False)

    for dataset, group in sample.groupby("dataset", sort=True):
        thumbs = []
        for _, item in group.sort_values("frame_number").iterrows():
            path = OVERLAYS / str(dataset) / f"{int(item.frame_number):08d}.jpg"
            image = cv2.imread(str(path))
            if image is None:
                continue
            thumb = cv2.resize(image, (192, 144), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, f"{int(item.frame_number)}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
            thumbs.append(thumb)
        cols = 6
        sheet = np.full((((len(thumbs) + cols - 1) // cols) * 144, cols * 192, 3), 245, dtype=np.uint8)
        for i, thumb in enumerate(thumbs):
            y, x = divmod(i, cols)
            sheet[y * 144:(y + 1) * 144, x * 192:(x + 1) * 192] = thumb
        contact = OUT / "contact_sheets" / f"{dataset}.jpg"
        contact.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(contact), sheet)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
