# Annotation sets

- `boxes_only_240.csv` is the latest 240-box file. The user confirms these coordinates were drawn manually; this is the annotation set used for the paper's secondary box audit. It contains one box per selected test frame and no `person_role_label` field.
- `manual_boxes_240.csv` is an earlier annotation file and is superseded by `boxes_only_240.csv` for the paper's latest results. Do not use its coordinates for the updated manual-box metrics.
- `assistant_insightface_face_boxes_240.csv` contains machine-generated InsightFace buffalo_l/SCRFD proposals for the same frames, clipped to the 800x600 image bounds. The coordinates exactly match `boxes_only_240.csv` across all 240 frames. The latest boxes remain user-designated manual annotations, but their agreement with these proposals is not an independent evaluation of InsightFace.
- `pseudo_boxes_yunet_scrfd_240.csv` is a separate machine-generated draft reference set. It averages YuNet and SCRFD boxes when the models agree and uses a YuNet proposal for 76 flagged frames. It is not human ground truth and must be described as pseudo-label evaluation.
- `ai_attacker_person_boxes_240.csv` contains machine-generated whole-person boxes. Its `attacker` class is assigned by user request and not image-verified; these are not face boxes.
- `qa_boxes_1.jpg` through `qa_boxes_4.jpg` show the InsightFace proposals overlaid on all 240 frames in sample order (60 frames per sheet).
- `tools/generate_insightface_face_boxes.py` recreates the InsightFace proposal file when the local source frames and buffalo_l weights are available.

A face box alone cannot establish whether a person is an attacker; that role needs temporal/action context and human verification.
