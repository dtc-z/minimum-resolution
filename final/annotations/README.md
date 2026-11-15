# Annotation sets

- `manual_boxes_240.csv` contains the user's 240 box coordinates and the corresponding landmark-proxy boxes. The `person_role_label` field is `attacker` by user assumption; it was not independently verified and is not an output of the face detectors.
- `pseudo_boxes_yunet_scrfd_240.csv` is a machine-generated draft reference set. It averages YuNet and SCRFD boxes when the models agree and uses the YuNet proposal for 76 flagged frames. It is not human ground truth and must be described as pseudo-label evaluation.

The pseudo set covers all 240 frames without local image paths. A face box alone cannot establish whether a person is an attacker; that role needs temporal/action context and human verification.
