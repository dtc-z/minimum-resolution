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


def frame_key(dataset, frame_number):
    return f"{dataset}:{int(frame_number)}"


def prediction_key(path):
    parts = str(path).replace("\\", "/").split("/")
    return f"{parts[-2]}:{int(Path(parts[-1]).stem)}"


def ap50(gt, predictions):
    ordered = predictions.sort_values("conf", ascending=False)
    matched = set()
    tp, fp = [], []
    for row in ordered.itertuples(index=False):
        key = row.frame_key
        box = [row.x1, row.y1, row.x2, row.y2]
        if key in gt and key not in matched and box_iou(box, gt[key]) >= 0.5:
            matched.add(key)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if not gt or not tp:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / len(gt)
    precision = ctp / np.maximum(ctp + cfp, 1)
    return float(np.mean([
        precision[recall >= r].max() if np.any(recall >= r) else 0.0
        for r in np.linspace(0, 1, 101)
    ]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final detector predictions against hand-drawn face boxes.")
    parser.add_argument("--annotations", required=True, help="CSV with dataset, frame_number, and human_x1..human_y2")
    parser.add_argument("--predictions", required=True, help="Final benchmark predictions CSV or CSV.GZ")
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", default="0.25,0.35,0.45,0.55,0.65,0.70")
    parser.add_argument("--ap-floor", type=float, default=0.001)
    args = parser.parse_args()

    ann = pd.read_csv(args.annotations)
    human_fields = ["dataset", "frame_number", "human_x1", "human_y1", "human_x2", "human_y2"]
    if set(human_fields) - set(ann.columns):
        raise ValueError(f"Annotation CSV missing fields: {sorted(set(human_fields) - set(ann.columns))}")
    ann = ann.dropna(subset=human_fields).drop_duplicates(["dataset", "frame_number"]).copy()
    ann["frame_key"] = [frame_key(d, n) for d, n in zip(ann.dataset, ann.frame_number)]
    paths = set(ann.frame_key)
    human_gt = {
        row.frame_key: [row.human_x1, row.human_y1, row.human_x2, row.human_y2]
        for row in ann.itertuples(index=False)
    }

    agreement_rows = []
    proxy_fields = ["proxy_x1", "proxy_y1", "proxy_x2", "proxy_y2"]
    if not (set(proxy_fields) - set(ann.columns)):
        for row in ann.itertuples(index=False):
            manual = human_gt[row.frame_key]
            proxy = [row.proxy_x1, row.proxy_y1, row.proxy_x2, row.proxy_y2]
            agreement_rows.append({
                "dataset": row.dataset,
                "frame_number": int(row.frame_number),
                "size_stratum": getattr(row, "size_stratum", ""),
                "manual_vs_proxy_iou": box_iou(manual, proxy),
                "manual_center_dx": ((manual[0] + manual[2]) - (proxy[0] + proxy[2])) / 2,
                "manual_center_dy": ((manual[1] + manual[3]) - (proxy[1] + proxy[3])) / 2,
                "manual_width_ratio": (manual[2] - manual[0]) / max(1e-9, proxy[2] - proxy[0]),
                "manual_height_ratio": (manual[3] - manual[1]) / max(1e-9, proxy[3] - proxy[1]),
            })

    required_predictions = ["model", "image_path", "resolution_width", "x1", "y1", "x2", "y2", "conf"]
    chunks = []
    rows_read = 0
    for chunk in pd.read_csv(args.predictions, usecols=required_predictions, chunksize=500_000):
        rows_read += len(chunk)
        chunk["frame_key"] = chunk.image_path.map(prediction_key)
        selected = chunk[chunk.frame_key.isin(paths) & (chunk.conf >= args.ap_floor)]
        if not selected.empty:
            chunks.append(selected.drop(columns="image_path"))
        if rows_read % 2_000_000 < 500_000:
            print(f"scanned {rows_read:,} prediction rows; retained {sum(len(item) for item in chunks):,}")
    if not chunks:
        raise ValueError("No predictions matched the annotated dataset/frame keys")
    preds = pd.concat(chunks, ignore_index=True)
    thresholds = [float(item) for item in args.thresholds.split(",")]
    records = []
    for (model, width), group in preds.groupby(["model", "resolution_width"], sort=True):
        ap = ap50(human_gt, group[group.conf >= args.ap_floor])
        for threshold in thresholds:
            selected = group[group.conf >= threshold]
            tp = fp = fn = 0
            for key, gt_box in human_gt.items():
                frame_preds = selected[selected.frame_key == key].sort_values("conf", ascending=False)
                hits = [box_iou([r.x1, r.y1, r.x2, r.y2], gt_box) >= 0.5
                        for r in frame_preds.itertuples(index=False)]
                if any(hits):
                    tp += 1
                    fp += max(0, len(hits) - 1)
                else:
                    fn += 1
                    fp += len(hits)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / len(paths) if paths else 0.0
            records.append({
                "model": model,
                "resolution_width": int(width),
                "conf_threshold": threshold,
                "ap50": ap,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "manual_images": len(paths),
                "reference_boxes": "user_drawn_manual",
            })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(out / "manual_box_summary.csv", index=False)
    stats = {
        "manual_images": len(paths),
        "models": sorted(preds.model.astype(str).unique().tolist()),
        "resolutions": sorted(int(v) for v in preds.resolution_width.unique()),
        "confidence_thresholds": thresholds,
        "ap_confidence_floor": args.ap_floor,
        "status": "computed_from_user_confirmed_hand_drawn_annotations_and_final_predictions",
    }
    if agreement_rows:
        agreement = pd.DataFrame(agreement_rows)
        agreement.to_csv(out / "box_agreement_by_frame.csv", index=False)
        stats["manual_vs_proxy_iou"] = agreement.manual_vs_proxy_iou.describe().to_dict()
        stats["manual_vs_proxy_iou_ge_0_5"] = float((agreement.manual_vs_proxy_iou >= 0.5).mean())
    (out / "manual_box_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
