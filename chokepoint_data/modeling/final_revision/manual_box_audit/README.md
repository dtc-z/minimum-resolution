# Manual face box audit

The candidate pool is generated only from the locked test manifest with seed 391. It contains 90 frames from each of the four test sequences, balanced across within-sequence eye-distance tertiles. Occlusion is human assessed because the source manifest has no occlusion field. After annotator A labels the candidates, the finalizer chooses 60 frames per sequence and 20 per size stratum while balancing the available occlusion classes. The final sample marks 48 frames for a second annotator.

1. Run `revision/sample_manual_audit.py` to create `manual_box_audit/screening_pool.csv` (360 candidates).
2. Annotator A labels every candidate and saves `screening_annotations_A.csv`.
3. Run `revision/finalize_manual_audit_sample.py` with the pool and screening annotations to create the final `sample.csv` (240 instances, balanced by sequence/size and stratified by assessed occlusion).
4. Annotator B labels only final rows marked `double_annotate` (48/240). Keep the original annotator files separate.
5. Draw the visible face extent, including visible facial area only when partially occluded. Do not use the orange derived box as a boundary guide; it is shown only for later comparison. Include forehead, cheeks, and chin; exclude hair and background. When a boundary is uncertain, use the visible skin/face contour and record the nearest consistent edge.
6. Resolve disagreements after independent annotation and enter adjudicated coordinates in a separate locked file. Preserve both original annotation files.

Launch the OpenCV UI with `revision/annotate_face_boxes.py --sample ... --annotator A|B --out ...`. Drag a rectangle, press `0`–`3` for occlusion, then Enter to save. Press `n` to skip and Esc to stop. The source data and sample are not modified.

Status: candidate sampler, occlusion-stratified finalizer, and tool are prepared; human annotation, adjudication, IoU/box sensitivity, and manual-label AP50/recall remain pending.

Workspace note: Windows ACLs deny ordinary processes read access to the source-image hardlinks. Training and benchmark runs in this workspace were launched with elevated read access; the annotation UI must also be able to read those source images. It writes annotation CSVs to the selected workspace output path and does not modify the source images.
