from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
COMPARISON = ROOT / "final/automated_audit_yunet/model_output/comparison_by_frame.csv"
FLAGS = ROOT / "final/automated_audit_yunet/model_output/review_flags.csv"
ROLES = ROOT / "final/annotations/manual_boxes_240.csv"
OUTPUT = ROOT / "final/annotations/pseudo_boxes_yunet_scrfd_240.csv"


def main() -> None:
    comparison = pd.read_csv(COMPARISON)
    flagged = pd.read_csv(FLAGS)
    role = pd.read_csv(ROLES)[["dataset", "frame_number", "person_role_label", "person_role_source"]]
    reasons = flagged.set_index(["dataset", "frame_number"])["audit_review_reason"].to_dict()

    rows = []
    for row in comparison.itertuples(index=False):
        key = (row.dataset, int(row.frame_number))
        reason = reasons.get(key, "")
        needs_review = bool(reason)
        if needs_review:
            box = [row.auto_target_x1_yunet, row.auto_target_y1_yunet,
                   row.auto_target_x2_yunet, row.auto_target_y2_yunet]
            method = "yunet_largest_face_fallback_needs_review"
        else:
            box = [
                (row.auto_target_x1_scrfd + row.auto_target_x1_yunet) / 2,
                (row.auto_target_y1_scrfd + row.auto_target_y1_yunet) / 2,
                (row.auto_target_x2_scrfd + row.auto_target_x2_yunet) / 2,
                (row.auto_target_y2_scrfd + row.auto_target_y2_yunet) / 2,
            ]
            method = "scrfd_yunet_detection_consensus"
        rows.append({
            "dataset": row.dataset,
            "frame_number": int(row.frame_number),
            "size_stratum": row.size_stratum_scrfd,
            "reference_x1": box[0], "reference_y1": box[1],
            "reference_x2": box[2], "reference_y2": box[3],
            "label_method": method,
            "scrfd_yunet_iou": row.model_box_iou_scrfd_vs_yunet,
            "confidence_scrfd": row.auto_target_score_scrfd,
            "confidence_yunet": row.auto_target_score_yunet,
            "review_required": needs_review,
            "review_reason": reason,
            "label_status": "machine_generated_unverified",
        })
    labels = pd.DataFrame(rows).merge(role, on=["dataset", "frame_number"], validate="one_to_one")
    if len(labels) != 240 or labels[["dataset", "frame_number"]].duplicated().any():
        raise ValueError("Expected 240 unique pseudo-box records")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(labels)} pseudo-boxes; {int(labels.review_required.sum())} require review.")


if __name__ == "__main__":
    main()
