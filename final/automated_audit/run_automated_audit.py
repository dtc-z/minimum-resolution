"""Compare automatic SCRFD proposals with user-drawn boxes on 240 test frames.

The reference boxes are hand-drawn by the user. SCRFD proposals remain automatic.
The script loads only InsightFace's detector module and does not run recognition.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from insightface.app import FaceAnalysis


ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "sample.csv"
OUT = ROOT / "model_output"
OVERLAYS = OUT / "overlays"
WEIGHTS = Path.home() / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx"
MODEL_NAME = "InsightFace buffalo_l / SCRFD-10GF detector only"
DETECTION_THRESHOLD = 0.10
DETECTION_SIZE = (640, 640)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def clipped_box(box, width: int, height: int) -> list[float]:
    return [
        float(np.clip(box[0], 0, width - 1)),
        float(np.clip(box[1], 0, height - 1)),
        float(np.clip(box[2], 0, width - 1)),
        float(np.clip(box[3], 0, height - 1)),
    ]


def draw_overlay(image, detections, chosen_box, chosen_score, reference_box, out_path: Path) -> None:
    canvas = image.copy()
    for box, score in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 255), 1)
        cv2.putText(canvas, f"SCRFD {score:.2f}", (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 180), 1, cv2.LINE_AA)
    px1, py1, px2, py2 = [int(round(v)) for v in reference_box]
    cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 0, 255), 2)
    cv2.putText(canvas, "USER-DRAWN BOX", (px1, max(16, py1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)
    if chosen_box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in chosen_box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(canvas, f"AUTO TARGET {chosen_score:.2f}", (x1, min(canvas.shape[0] - 8, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(SAMPLE)
    if len(sample) != 240:
        raise ValueError(f"Expected 240 audit samples, found {len(sample)}")
    if "box_source" not in sample or not sample.box_source.eq("user_drawn_manual_annotation").all():
        raise ValueError("The 240 reference boxes must be tagged as user-drawn manual annotations")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Expected pretrained detector weights at {WEIGHTS}")

    # ONNX Runtime CUDAExecutionProvider is installed, but this Windows
    # environment cannot load its cuDNN 9 dependency; use the working CPU EP.
    providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=providers)
    app.prepare(ctx_id=0, det_size=DETECTION_SIZE, det_thresh=DETECTION_THRESHOLD)

    rows = []
    for index, item in sample.iterrows():
        image_path = Path(item["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        row = item.to_dict()
        row.update({
            "automated_detector": MODEL_NAME,
            "detection_threshold": DETECTION_THRESHOLD,
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

        height, width = image.shape[:2]
        try:
            faces = app.get(image)
            detections = []
            for face in faces:
                box = clipped_box(face.bbox, width, height)
                score = float(face.det_score)
                area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                detections.append({"box": box, "score": score, "area": area})
            row["face_count"] = len(detections)
            row["all_detections_json"] = json.dumps(
                [{"box": d["box"], "score": d["score"]} for d in detections],
                separators=(",", ":"),
            )
            chosen = max(detections, key=lambda d: d["area"]) if detections else None
            if chosen:
                box = chosen["box"]
                score = chosen["score"]
                row.update({
                    "auto_target_x1": box[0], "auto_target_y1": box[1],
                    "auto_target_x2": box[2], "auto_target_y2": box[3],
                    "auto_target_score": score,
                    "iou_auto_vs_manual_box": iou(
                        box, [item["x1"], item["y1"], item["x2"], item["y2"]]
                    ),
                    "auto_center_distance_px": float(np.hypot(
                        (box[0] + box[2] - item["x1"] - item["x2"]) / 2,
                        (box[1] + box[3] - item["y1"] - item["y2"]) / 2,
                    )),
                    "auto_to_manual_area_ratio": (
                        max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                        / max(1.0, (item["x2"] - item["x1"]) * (item["y2"] - item["y1"]))
                    ),
                })
            overlay = OVERLAYS / str(item["dataset"]) / f"{int(item['frame_number']):08d}.jpg"
            draw_overlay(image, [(d["box"], d["score"]) for d in detections],
                         chosen["box"] if chosen else None,
                         chosen["score"] if chosen else None,
                         [item["x1"], item["y1"], item["x2"], item["y2"]], overlay)
        except Exception as exc:  # preserve a row for every sample if inference fails
            row["image_read_status"] = "inference_error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if (index + 1) % 10 == 0:
            print(f"processed {index + 1}/240")

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "model_proposals.csv", index=False, encoding="utf-8-sig")
    successful = result[result["auto_target_score"].notna()]
    providers_used = list(ort.get_available_providers())
    execution_providers = app.models["detection"].session.get_providers()
    metadata = {
        "status": "automated_scrfd_proposals_compared_with_user_drawn_manual_boxes",
        "human_annotation": True,
        "annotation_source": "user_drawn_manual_annotation",
        "reference_box_source": "user_drawn_manual_annotation",
        "paper_modified": False,
        "occlusion": "assumed_none based on user-provided statement; not independently assessed",
        "sample_count": len(sample),
        "sample_seed": 391,
        "sample_rule": "20 per test sequence x eye-distance size stratum; 4 sequences x 3 strata",
        "target_box_rule": "largest detected face box in each frame; all detector boxes retained in CSV",
        "model": MODEL_NAME,
        "weights": str(WEIGHTS),
        "weights_sha256": sha256(WEIGHTS),
        "license": "InsightFace public pretrained models: non-commercial research only",
        "allowed_modules": ["detection"],
        "detector_threshold": DETECTION_THRESHOLD,
        "detector_input_size": DETECTION_SIZE,
        "onnxruntime_providers_available": providers_used,
        "onnxruntime_execution_providers": execution_providers,
        "torch_cuda": torch.cuda.is_available(),
        "inference_device": "GPU" if "CUDAExecutionProvider" in execution_providers else "CPU",
        "available_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "images_with_detection": len(successful),
        "images_without_detection": int((result["auto_target_score"].isna()).sum()),
        "mean_model_confidence": float(successful["auto_target_score"].mean()) if len(successful) else None,
        "median_iou_with_manual_boxes": float(successful["iou_auto_vs_manual_box"].median()) if len(successful) else None,
        "median_center_distance_px": float(successful["auto_center_distance_px"].median()) if len(successful) else None,
        "median_auto_to_manual_area_ratio": float(successful["auto_to_manual_area_ratio"].median()) if len(successful) else None,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Compact per-sequence/face-size summaries for quick review.
    successful.groupby(["dataset", "size_stratum"], dropna=False).agg(
        samples=("image_path", "count"),
        median_iou=("iou_auto_vs_manual_box", "median"),
        median_center_distance_px=("auto_center_distance_px", "median"),
        median_auto_to_manual_area_ratio=("auto_to_manual_area_ratio", "median"),
        median_confidence=("auto_target_score", "median"),
    ).reset_index().to_csv(OUT / "summary_by_sequence_and_size.csv", index=False)

    # Contact sheets show both boxes: red is the user-drawn reference, green the
    # model's largest detection. They are visual model-assisted review aids.
    for dataset, group in sample.groupby("dataset", sort=True):
        thumbs = []
        for _, item in group.sort_values("frame_number").iterrows():
            path = OVERLAYS / str(dataset) / f"{int(item['frame_number']):08d}.jpg"
            image = cv2.imread(str(path))
            if image is None:
                continue
            thumb = cv2.resize(image, (192, 144), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, f"{int(item['frame_number'])}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
            thumbs.append(thumb)
        cols = 6
        rows_sheet = (len(thumbs) + cols - 1) // cols
        sheet = np.full((rows_sheet * 144, cols * 192, 3), 245, dtype=np.uint8)
        for i, thumb in enumerate(thumbs):
            y, x = divmod(i, cols)
            sheet[y * 144:(y + 1) * 144, x * 192:(x + 1) * 192] = thumb
        sheet_path = OUT / "contact_sheets" / f"{dataset}.jpg"
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet_path), sheet)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
