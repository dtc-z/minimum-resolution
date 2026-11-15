"""Evaluate final detector predictions against the latest InsightFace proposal set.

The reference boxes are machine generated, not human ground truth. Results are
descriptive agreement scores for a 240-frame subset only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "annotations" / "assistant_insightface_face_boxes_240.csv"
PREDICTIONS = ROOT.parent / "chokepoint_data/modeling/final_revision/benchmark/predictions.csv.gz"
OUT = ROOT / "benchmark" / "insightface_reference_audit"
THRESHOLDS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.70]
AP_FLOOR = 0.001


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def get_frame_key(dataset, frame):
    return f"{dataset}:{int(frame)}"


def prediction_frame_key(image_path):
    parts = str(image_path).replace("\\", "/").split("/")
    return f"{parts[-2]}:{int(Path(parts[-1]).stem)}"


def ap50(gt, predictions):
    ordered = predictions.sort_values("conf", ascending=False)
    matched, true_positives, false_positives = set(), [], []
    for row in ordered.itertuples(index=False):
        key = row.frame_key
        box = [row.x1, row.y1, row.x2, row.y2]
        hit = key in gt and key not in matched and box_iou(box, gt[key]) >= 0.5
        if hit:
            matched.add(key)
        true_positives.append(int(hit))
        false_positives.append(int(not hit))
    if not true_positives:
        return 0.0
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recall = cumulative_tp / len(gt)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    return float(np.mean([
        precision[recall >= level].max() if np.any(recall >= level) else 0.0
        for level in np.linspace(0, 1, 101)
    ]))


def main():
    labels = pd.read_csv(REFERENCE)
    required = ["dataset", "frame_number", "x1", "y1", "x2", "y2", "label_source"]
    missing = sorted(set(required) - set(labels.columns))
    if missing:
        raise ValueError(f"Reference CSV lacks columns: {missing}")
    if labels.duplicated(["dataset", "frame_number"]).any():
        raise ValueError("Reference CSV has duplicate dataset/frame keys")
    if not labels["label_source"].astype(str).str.contains("machine_generated", case=False).all():
        raise ValueError("Expected the current machine-generated InsightFace reference set")
    labels["frame_key"] = [get_frame_key(d, n) for d, n in zip(labels.dataset, labels.frame_number)]
    gt = {r.frame_key: [r.x1, r.y1, r.x2, r.y2] for r in labels.itertuples(index=False)}
    keys = set(gt)

    needed = ["model", "image_path", "resolution_width", "x1", "y1", "x2", "y2", "conf"]
    chunks, rows_read, rows_kept = [], 0, 0
    for chunk in pd.read_csv(PREDICTIONS, usecols=needed, chunksize=500_000):
        rows_read += len(chunk)
        chunk["frame_key"] = chunk["image_path"].map(prediction_frame_key)
        selected = chunk[chunk.frame_key.isin(keys) & (chunk.conf >= AP_FLOOR)].drop(columns="image_path")
        if not selected.empty:
            chunks.append(selected)
            rows_kept += len(selected)
        if rows_read % 2_000_000 < 500_000:
            print(f"Scanned {rows_read:,}; retained {rows_kept:,} reference-frame predictions")
    if not chunks:
        raise ValueError("No final predictions matched the reference frames")
    predictions = pd.concat(chunks, ignore_index=True)

    results = []
    for (model, width), group in predictions.groupby(["model", "resolution_width"], sort=True):
        ap = ap50(gt, group)
        for threshold in THRESHOLDS:
            selected = group[group.conf >= threshold]
            tp = fp = fn = 0
            by_frame = {key: rows for key, rows in selected.groupby("frame_key")}
            for key, target in gt.items():
                rows = by_frame.get(key)
                if rows is None:
                    fn += 1
                    continue
                hits = [box_iou([r.x1, r.y1, r.x2, r.y2], target) >= 0.5
                        for r in rows.sort_values("conf", ascending=False).itertuples(index=False)]
                if any(hits):
                    tp += 1
                    fp += len(hits) - 1
                else:
                    fn += 1
                    fp += len(hits)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / len(gt)
            results.append({
                "model": model,
                "resolution_width": int(width),
                "conf_threshold": threshold,
                "ap50": ap,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "tp": tp, "fp": fp, "fn": fn,
                "reference_frames": len(gt),
                "reference_source": "InsightFace buffalo_l/SCRFD; machine generated, not human ground truth",
            })

    OUT.mkdir(parents=True, exist_ok=True)
    frame_summary = pd.DataFrame(results)
    frame_summary.to_csv(OUT / "insightface_reference_summary.csv", index=False)
    metadata = {
        "reference_file": str(REFERENCE.relative_to(ROOT)),
        "reference_rows": len(labels),
        "reference_model": "InsightFace buffalo_l/SCRFD proposal",
        "human_ground_truth": False,
        "prediction_rows_read": rows_read,
        "prediction_rows_retained": rows_kept,
        "ap_confidence_floor": AP_FLOOR,
        "operating_thresholds": THRESHOLDS,
        "match_iou": 0.5,
        "status": "exploratory_model_generated_reference_agreement",
        "note": "Do not interpret as human-label accuracy or independent ground truth.",
    }
    (OUT / "insightface_reference_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
