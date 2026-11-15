"""Compare YuNet and SCRFD outputs on the exact same 240 audit frames."""

from pathlib import Path

import numpy as np
import pandas as pd
import cv2


ROOT = Path(__file__).resolve().parent
SCRFD = Path(__file__).resolve().parents[1] / "automated_audit" / "model_output" / "model_proposals.csv"
YUNET = ROOT / "model_output" / "model_proposals.csv"


def box_iou(row, left: str, right: str) -> float:
    a = [row[f"auto_target_{axis}_{left}"] for axis in ("x1", "y1", "x2", "y2")]
    b = [row[f"auto_target_{axis}_{right}"] for axis in ("x1", "y1", "x2", "y2")]
    if any(pd.isna(value) for value in a + b):
        return np.nan
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def main() -> None:
    a = pd.read_csv(SCRFD)
    b = pd.read_csv(YUNET)
    keys = ["dataset", "frame_number"]
    columns = keys + ["size_stratum", "x1", "y1", "x2", "y2", "face_count", "auto_target_score", "iou_auto_vs_manual_box",
                      "auto_center_distance_px", "auto_to_manual_area_ratio",
                      "auto_target_x1", "auto_target_y1", "auto_target_x2", "auto_target_y2"]
    comparison = a[columns].merge(b[columns], on=keys, suffixes=("_scrfd", "_yunet"), validate="one_to_one")
    comparison["person_role_label"] = "attacker"
    comparison["person_role_source"] = "user_assumption"
    comparison["same_manual_boxes"] = (
        (comparison.x1_scrfd - comparison.x1_yunet).abs().lt(1e-5)
        & (comparison.y1_scrfd - comparison.y1_yunet).abs().lt(1e-5)
        & (comparison.x2_scrfd - comparison.x2_yunet).abs().lt(1e-5)
        & (comparison.y2_scrfd - comparison.y2_yunet).abs().lt(1e-5)
    )
    comparison["model_box_iou_scrfd_vs_yunet"] = comparison.apply(
        box_iou, axis=1, args=("scrfd", "yunet")
    )
    comparison["iou_difference_yunet_minus_scrfd"] = (
        comparison.iou_auto_vs_manual_box_yunet - comparison.iou_auto_vs_manual_box_scrfd
    )
    comparison["center_distance_difference_yunet_minus_scrfd_px"] = (
        comparison.auto_center_distance_px_yunet - comparison.auto_center_distance_px_scrfd
    )
    comparison["both_detected"] = (
        comparison.auto_target_score_scrfd.notna() & comparison.auto_target_score_yunet.notna()
    )
    comparison["audit_review_reason"] = ""
    low_reference_overlap = comparison.iou_auto_vs_manual_box_yunet.lt(0.50)
    low_model_agreement = comparison.model_box_iou_scrfd_vs_yunet.lt(0.50)
    multiple_yunet = comparison.face_count_yunet.gt(1)
    multiple_scrfd = comparison.face_count_scrfd.gt(1)
    comparison.loc[low_reference_overlap, "audit_review_reason"] += "yunet_reference_iou_below_0.50;"
    comparison.loc[low_model_agreement, "audit_review_reason"] += "model_box_iou_below_0.50;"
    comparison.loc[multiple_yunet, "audit_review_reason"] += "multiple_yunet_detections;"
    comparison.loc[multiple_scrfd, "audit_review_reason"] += "multiple_scrfd_detections;"
    review_queue = comparison[comparison.audit_review_reason.ne("")].copy()

    rows = []
    for stratum, group in [("all", comparison), *comparison.groupby("size_stratum_scrfd", sort=True)]:
        n = len(group)
        for model in ("scrfd", "yunet"):
            detected = group[f"auto_target_score_{model}"].notna()
            valid = group.loc[detected]
            rows.append({
                "stratum": stratum,
                "model": model,
                "person_role_label": "attacker",
                "person_role_source": "user_assumption",
                "frames": n,
                "detections": int(detected.sum()),
                "detection_rate": float(detected.mean()),
                "mean_confidence": float(valid[f"auto_target_score_{model}"].mean()) if len(valid) else np.nan,
                "median_iou_vs_manual_box": float(valid[f"iou_auto_vs_manual_box_{model}"].median()) if len(valid) else np.nan,
                "median_center_distance_px": float(valid[f"auto_center_distance_px_{model}"].median()) if len(valid) else np.nan,
                "median_area_ratio_to_manual_box": float(valid[f"auto_to_manual_area_ratio_{model}"].median()) if len(valid) else np.nan,
            })

    OUT = ROOT / "model_output"
    comparison.to_csv(OUT / "comparison_by_frame.csv", index=False, encoding="utf-8-sig")
    review_queue.to_csv(OUT / "review_flags.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "comparison_summary.csv", index=False)
    paired = comparison[comparison.both_detected]
    paired_summary = {
        "same_frame_count": len(comparison),
        "same_manual_reference_boxes": bool(comparison.same_manual_boxes.all()),
        "person_role_assumption": "attacker",
        "person_role_source": "user_assumption; not classified by the detectors",
        "frames_detected_by_both": len(paired),
        "frames_flagged_for_visual_review": len(review_queue),
        "median_iou_between_yunet_and_scrfd_boxes": float(paired.model_box_iou_scrfd_vs_yunet.median()) if len(paired) else None,
        "median_iou_gain_yunet_minus_scrfd_vs_manual": float(paired.iou_difference_yunet_minus_scrfd.median()) if len(paired) else None,
        "median_center_distance_change_yunet_minus_scrfd_px": float(paired.center_distance_difference_yunet_minus_scrfd_px.median()) if len(paired) else None,
    }
    import json
    (OUT / "comparison_metadata.json").write_text(json.dumps(paired_summary, indent=2), encoding="utf-8")
    for dataset, group in review_queue.groupby("dataset", sort=True):
        thumbs = []
        for _, row in group.sort_values("frame_number").iterrows():
            overlay_path = OUT / "overlays" / str(dataset) / f"{int(row.frame_number):08d}.jpg"
            image = cv2.imread(str(overlay_path))
            if image is None:
                continue
            thumb = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, f"{int(row.frame_number)} IoU {row.iou_auto_vs_manual_box_yunet:.2f}",
                        (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
            thumbs.append(thumb)
        cols = 4
        sheet = np.full((((len(thumbs) + cols - 1) // cols) * 240, cols * 320, 3), 245, dtype=np.uint8)
        for i, thumb in enumerate(thumbs):
            y, x = divmod(i, cols)
            sheet[y * 240:(y + 1) * 240, x * 320:(x + 1) * 320] = thumb
        contact = OUT / "review_contact_sheets" / f"{dataset}.jpg"
        contact.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(contact), sheet)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(paired_summary, indent=2))


if __name__ == "__main__":
    main()
