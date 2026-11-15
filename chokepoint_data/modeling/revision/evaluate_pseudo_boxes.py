from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter) if aa + ab > inter else 0.0


def key(dataset, frame):
    return f"{dataset}:{int(frame)}"


def key_from_image(path):
    parts = str(path).replace("\\", "/").split("/")
    return f"{parts[-2]}:{int(Path(parts[-1]).stem)}"


def ap50(gt, preds):
    ordered = preds.sort_values("conf", ascending=False)
    seen, tp, fp = set(), [], []
    for row in ordered.itertuples(index=False):
        box = [row.x1, row.y1, row.x2, row.y2]
        if row.frame_key in gt and row.frame_key not in seen and iou(box, gt[row.frame_key]) >= 0.5:
            seen.add(row.frame_key); tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    if not tp:
        return 0.0
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall = tp / len(gt); precision = tp / np.maximum(tp + fp, 1)
    return float(np.mean([precision[recall >= r].max() if np.any(recall >= r) else 0.0
                          for r in np.linspace(0, 1, 101)]))


def main():
    parser = argparse.ArgumentParser(description="Evaluate detectors against model-generated pseudo boxes; not human ground truth.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ap-floor", type=float, default=0.001)
    parser.add_argument("--thresholds", default="0.25,0.35,0.45,0.55,0.65,0.70")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    required = {"dataset", "frame_number", "reference_x1", "reference_y1", "reference_x2", "reference_y2"}
    if required - set(labels.columns):
        raise ValueError(f"Pseudo-label CSV missing columns: {sorted(required-set(labels.columns))}")
    labels = labels.drop_duplicates(["dataset", "frame_number"])
    labels["frame_key"] = [key(d, n) for d, n in zip(labels.dataset, labels.frame_number)]
    gt = {r.frame_key: [r.reference_x1, r.reference_y1, r.reference_x2, r.reference_y2]
          for r in labels.itertuples(index=False)}

    columns = ["model", "image_path", "resolution_width", "x1", "y1", "x2", "y2", "conf"]
    kept = []; scanned = 0
    for chunk in pd.read_csv(args.predictions, usecols=columns, chunksize=500_000):
        scanned += len(chunk)
        chunk["frame_key"] = chunk.image_path.map(key_from_image)
        take = chunk[chunk.frame_key.isin(gt) & (chunk.conf >= args.ap_floor)]
        if not take.empty:
            kept.append(take.drop(columns="image_path"))
        if scanned % 2_000_000 < 500_000:
            print(f"scanned {scanned:,}; retained {sum(len(x) for x in kept):,}")
    preds = pd.concat(kept, ignore_index=True)
    thresholds = [float(v) for v in args.thresholds.split(",")]
    rows = []
    for (model, width), group in preds.groupby(["model", "resolution_width"], sort=True):
        ap = ap50(gt, group[group.conf >= args.ap_floor])
        for threshold in thresholds:
            selected = group[group.conf >= threshold]
            tp = fp = fn = 0
            for frame_key, ref in gt.items():
                boxes = selected[selected.frame_key == frame_key].sort_values("conf", ascending=False)
                hits = [iou([r.x1, r.y1, r.x2, r.y2], ref) >= 0.5 for r in boxes.itertuples(index=False)]
                if any(hits): tp += 1; fp += len(hits) - 1
                else: fn += 1; fp += len(hits)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / len(gt) if gt else 0.0
            rows.append({"model": model, "resolution_width": int(width), "conf_threshold": threshold,
                         "ap50": ap, "precision": precision, "recall": recall,
                         "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
                         "tp": tp, "fp": fp, "fn": fn, "pseudo_images": len(gt),
                         "reference_type": "yunet_scrfd_pseudo_boxes"})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows); result.to_csv(out / "pseudo_box_summary.csv", index=False)
    metadata = {"pseudo_images": len(gt), "source": "YuNet/SCRFD model proposals",
                "ap_confidence_floor": args.ap_floor, "confidence_thresholds": thresholds,
                "test_rows_scanned": scanned, "retained_rows": len(preds),
                "not_human_ground_truth": True,
                "status": "descriptive_model_consistency_against_pseudo_boxes"}
    (out / "pseudo_box_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
