# Manual face-box audit

The 240 reference boxes in `../annotations/manual_boxes_240.csv` were drawn by the user. The earlier `box_source=automatic_SCRFD_10GF_proposal` value was assigned incorrectly. The annotation file keeps sequence/frame keys and box coordinates without local image paths.

`model_output/` contains a separate automated SCRFD-10GF proposal pass against those hand-drawn reference boxes. It ran the detector only; no face recognition or embeddings were used. The detector selected its largest face proposal in each image. The 240 annotations are balanced across four test sequences and three eye-distance strata (20 frames per stratum in each sequence).

## Results

The automatic proposals were detected on 240/240 frames. Mean detector confidence was 0.857; median IoU with the user-drawn boxes was 0.624; median center distance was 13.33 px; median proposal-to-reference area ratio was 0.646. The user-drawn boxes exactly match the corresponding landmark-proxy coordinates (IoU 1.000 for all 240 frames), so this audit does not independently estimate proxy-label bias. This proposal pass is separate from the four fine-tuned detectors in the main benchmark.

## Files

- `../annotations/manual_boxes_240.csv` — hand-drawn reference coordinates, keyed by sequence and frame number.
- `create_sample.py` — creates a candidate list from the landmark-proxy manifest; it does not overwrite the manual annotation file.
- `run_automated_audit.py` — SCRFD proposals and compact comparison summaries.
- `model_output/summary_by_sequence_and_size.csv` — aggregate proposal comparison by sequence and face-size stratum.
- `model_output/run_metadata.json` — model, runtime, and annotation-source metadata.
- Local per-frame detector outputs and visual overlays are excluded from version control because they contain absolute image paths or face frames.

The pretrained InsightFace weights are licensed for non-commercial research. See the [official SCRFD results](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/README.md) and [model zoo terms](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md).

## Independent-architecture comparison

A second automatic proposal pass used OpenCV Zoo YuNet, which is architecturally distinct from the benchmark detectors. It detected faces in 240/240 of the same frames, with median IoU 0.628 and median center distance 11.28 px against the user-drawn boxes. On those same frames, SCRFD's values were 0.624 and 13.33 px. The median IoU between the two models' selected boxes was 0.911. Seventy-six frames are flagged for closer inspection due to multiple detections or lower box agreement. Full results and overlays are in [`../automated_audit_yunet/`](../automated_audit_yunet/). Both model passes remain proposals, not human annotations; the reference boxes still match the landmark proxies exactly, so neither comparison independently estimates proxy-label bias. The paper and reference annotations were not modified.

For the 240 audit frames, the target's `person_role_label` is set to `attacker` under the user's assumption; neither face detector infers that role.
