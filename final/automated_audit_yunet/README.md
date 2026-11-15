# YuNet model-assisted box audit

This is a second automatic face-box proposal pass on the same fixed 240-frame sample used for SCRFD. YuNet is from the [OpenCV Zoo face-detection model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet); its published model files are MIT-licensed. The run used OpenCV 4.11.0 on CPU, input 640×640, score threshold 0.30, and NMS threshold 0.30. Input images were aspect-preserving resized and centered on a black square; detections were mapped back to the original frame coordinates.

The subject's `person_role_label` is set to `attacker` on the user's assumption for these 240 audit frames. YuNet detects faces only; it does not classify who is hitting whom, and the role was not independently verified.

## Results

YuNet detected a face in 240/240 frames. Median IoU against the user-drawn reference boxes was 0.628, median center distance 11.28 px, and median model-box/reference area ratio 0.641. On the same frames, SCRFD achieved median IoU 0.624 and median center distance 13.33 px. The median IoU between the two models' selected boxes was 0.911. Detector confidence values are model-specific and should not be compared across models.

The 240 reference boxes equal the existing landmark-proxy boxes exactly, so agreement with these references does not independently measure any bias in the original proxy labels. The no-occlusion condition is the user's assumption and was not independently assessed. YuNet/SCRFD boxes are used only in the paper's explicitly labeled exploratory pseudo-label sensitivity check; the detectors do not verify the `attacker` role.

## Files

- `model_output/model_proposals.csv` — all YuNet detections and per-frame comparison fields.
- `model_output/run_metadata.json` — model, checkpoint hash, settings, provenance, and overall metrics.
- `model_output/summary_by_sequence_and_size.csv` — results per sequence and eye-distance stratum.
- `model_output/comparison_by_frame.csv` — side-by-side SCRFD and YuNet boxes for each frame.
- `model_output/comparison_summary.csv` and `comparison_metadata.json` — overall and size-stratified comparison.
- `model_output/review_flags.csv` — 76 candidates flagged for closer inspection: YuNet/reference IoU below 0.50, inter-model IoU below 0.50, or multiple face detections in either model.
- `model_output/review_contact_sheets/` — larger overlays for those flagged frames, grouped by sequence.
- `model_output/overlays/` — one overlay per frame; red is the user-drawn reference, green the largest YuNet box, yellow other YuNet boxes.
- `model_output/contact_sheets/` — visual summaries by sequence.
- `run_yunet_audit.py` and `compare_models.py` — reproducible inference and comparison scripts.

The automatic boxes are model proposals, not human annotations. No identity-recognition module was used.
