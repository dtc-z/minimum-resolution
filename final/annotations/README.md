# Annotation sets

- `manual_boxes_240.csv` contains the user's 240 box coordinates and the corresponding landmark-proxy boxes. The `person_role_label` field is `attacker` by user assumption; it was not independently verified and is not an output of the face detectors.
- `pseudo_boxes_yunet_scrfd_240.csv` is a machine-generated draft reference set. It averages YuNet and SCRFD boxes when the models agree and uses the YuNet proposal for 76 flagged frames. It is not human ground truth and must be described as pseudo-label evaluation.

- `assistant_insightface_face_boxes_240.csv` contains one face box for each selected test frame. Boxes were generated locally with InsightFace buffalo_l/SCRFD and clipped to the 800x600 image bounds. This is a machine-generated annotation set, not a human label set or independent ground truth. The `attacker` class value is assigned at the user's request; it is not verified from the image. Use it as a detector proposal or draft annotation set, not as human-labeled test ground truth.
- `qa_boxes_1.jpg` through `qa_boxes_4.jpg` show the boxes overlaid on all 240 frames in sample order (60 frames per sheet).
- `tools/generate_insightface_face_boxes.py` recreates the InsightFace box file when the local source frames and buffalo_l weights are available.

The pseudo set covers all 240 frames without local image paths. A face box alone cannot establish whether a person is an attacker; that role needs temporal/action context and human verification.
