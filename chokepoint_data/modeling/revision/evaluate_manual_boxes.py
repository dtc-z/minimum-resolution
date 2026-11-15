from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def ap50(gt, predictions):
    ordered = predictions.sort_values("conf", ascending=False)
    matched = set()
    tp, fp = [], []
    for row in ordered.itertuples(index=False):
        path = row.image_path
        box = [row.x1, row.y1, row.x2, row.y2]
        if path in gt and path not in matched and box_iou(box, gt[path]) >= 0.5:
            matched.add(path)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if not gt:
        return 0.0
    if not tp:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / len(gt)
    precision = ctp / np.maximum(ctp + cfp, 1)
    return float(np.mean([precision[recall >= r].max() if np.any(recall >= r) else 0.0
                          for r in np.linspace(0, 1, 101)]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare detector outputs against adjudicated human face rectangles.")
    parser.add_argument("--annotations", required=True, help="Adjudicated CSV with one human box per image_path")
    parser.add_argument("--predictions", required=True, help="Final benchmark predictions CSV or CSV.GZ")
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", default="0.25,0.35,0.45,0.55,0.65,0.70")
    args = parser.parse_args()

    ann = pd.read_csv(args.annotations)
    preds = pd.read_csv(args.predictions)
    human_fields = ["image_path", "human_x1", "human_y1", "human_x2", "human_y2"]
    required_predictions = {"model", "resolution_width", "image_path", "x1", "y1", "x2", "y2", "conf"}
    if set(human_fields) - set(ann.columns) or required_predictions - set(preds.columns):
        raise ValueError("Missing human annotation or prediction columns")
    ann = ann.dropna(subset=human_fields).drop_duplicates("image_path")
    ann["image_path"] = ann.image_path.astype(str)
    paths = set(ann.image_path)
    preds = preds[preds.image_path.astype(str).isin(paths)].copy()
    thresholds = [float(item) for item in args.thresholds.split(",")]
    human_gt = {row.image_path: [row.human_x1, row.human_y1, row.human_x2, row.human_y2]
                for row in ann.itertuples(index=False)}
    derived = {row.image_path: [row.x1, row.y1, row.x2, row.y2] for row in ann.itertuples(index=False)}

    box_audit = []
    for row in ann.itertuples(index=False):
        d = derived[row.image_path]
        h = human_gt[row.image_path]
        box_audit.append({"image_path": row.image_path, "dataset": row.dataset,
                          "eye_distance_px": row.eye_distance_px, "size_stratum": row.size_stratum,
                          "human_occlusion": row.human_occlusion, "derived_human_iou": box_iou(d, h),
                          "derived_center_dx": ((d[0] + d[2]) - (h[0] + h[2])) / 2,
                          "derived_center_dy": ((d[1] + d[3]) - (h[1] + h[3])) / 2,
                          "derived_width_ratio": (d[2] - d[0]) / max(1e-9, h[2] - h[0]),
                          "derived_height_ratio": (d[3] - d[1]) / max(1e-9, h[3] - h[1])})
    box_audit = pd.DataFrame(box_audit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    box_audit.to_csv(out / "box_agreement.csv", index=False)

    records = []
    for factor in (0.9, 1.0, 1.1):
        scaled_gt = {}
        for path, box in derived.items():
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            scaled_gt[path] = [max(0, cx - (cx - box[0]) * factor),
                               max(0, cy - (cy - box[1]) * factor),
                               min(799, cx + (box[2] - cx) * factor),
                               min(599, cy + (box[3] - cy) * factor)]
        for (model, width), group in preds.groupby(["model", "resolution_width"], sort=True):
            for threshold in thresholds:
                selected = group[group.conf >= threshold]
                counts = {}
                for image_path in paths:
                    frame_preds = selected[selected.image_path.astype(str) == image_path].sort_values("conf", ascending=False)
                    counts[image_path] = (frame_preds, scaled_gt[image_path])
                tp = fp = fn = 0
                for frame_preds, gt_box in counts.values():
                    hits = [box_iou([r.x1, r.y1, r.x2, r.y2], gt_box) >= 0.5 for r in frame_preds.itertuples(index=False)]
                    if any(hits):
                        tp += 1
                        fp += max(0, len(hits) - 1)
                    else:
                        fn += 1
                        fp += len(hits)
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / len(paths) if paths else 0.0
                records.append({"model": model, "resolution_width": int(width), "conf_threshold": threshold,
                                "reference_boxes": f"derived_scaled_{factor:.1f}", "ap50": ap50(scaled_gt, selected),
                                "precision": precision, "recall": recall,
                                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                                "tp": tp, "fp": fp, "fn": fn, "manual_images": len(paths)})
    # A separate evaluation against adjudicated manual rectangles.
    for (model, width), group in preds.groupby(["model", "resolution_width"], sort=True):
        for threshold in thresholds:
            selected = group[group.conf >= threshold]
            tp = fp = fn = 0
            for image_path, gt_box in human_gt.items():
                frame_preds = selected[selected.image_path.astype(str) == image_path].sort_values("conf", ascending=False)
                hits = [box_iou([r.x1, r.y1, r.x2, r.y2], gt_box) >= 0.5 for r in frame_preds.itertuples(index=False)]
                if any(hits):
                    tp += 1
                    fp += max(0, len(hits) - 1)
                else:
                    fn += 1
                    fp += len(hits)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / len(paths) if paths else 0.0
            records.append({"model": model, "resolution_width": int(width), "conf_threshold": threshold,
                            "reference_boxes": "manual_adjudicated", "ap50": ap50(human_gt, selected),
                            "precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                            "tp": tp, "fp": fp, "fn": fn, "manual_images": len(paths)})
    result = pd.DataFrame(records)
    result.to_csv(out / "manual_box_summary.csv", index=False)
    stats = {"manual_images": len(paths), "derived_vs_manual_iou": box_audit.derived_human_iou.describe().to_dict(),
             "models": sorted(preds.model.astype(str).unique().tolist()), "resolutions": sorted(preds.resolution_width.unique().tolist()),
             "status": "computed_from_adjudicated_annotations_and_final_predictions"}
    (out / "manual_box_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
