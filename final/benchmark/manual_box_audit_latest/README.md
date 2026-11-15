# Latest manual box audit

The source reference is `../../annotations/boxes_only_240.csv`, the latest 240-box file that the user confirms was drawn manually. The earlier `manual_boxes_240.csv` is superseded for the paper.

`boxes_for_evaluation.csv` joins those boxes to the landmark-proxy boxes by dataset and frame. The final detector prediction archive was evaluated for four models, ten input widths, and six confidence thresholds using the existing manual-box evaluation protocol. Results are in `manual_box_summary.csv`; per-frame proxy agreement is in `box_agreement_by_frame.csv`; aggregate counts and IoU statistics are in `manual_box_summary.json`.

Across 240 frames, manual-to-proxy IoU has mean 0.625, median 0.625, and IQR 0.593 to 0.654; 239 of 240 pairs reach IoU 0.50. The `boxes_only_240.csv` coordinates exactly match `assistant_insightface_face_boxes_240.csv` on all 240 frames. The paper treats the boxes as user-designated manual annotations, while disclosing that the comparison is not independent validation of the InsightFace proposals.
