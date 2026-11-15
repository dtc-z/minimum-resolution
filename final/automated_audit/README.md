# Historical automated proposal audit

This folder records a proposal audit against the earlier `../annotations/manual_boxes_240.csv` file. That annotation file is superseded by the latest user-confirmed hand-drawn `../annotations/boxes_only_240.csv`; this historical audit is not the manual-box result reported in the paper. The earlier `box_source=automatic_SCRFD_10GF_proposal` value was assigned incorrectly at the time. The annotation file keeps sequence/frame keys and box coordinates without local image paths.

`model_output/` contains a separate automated SCRFD-10GF proposal pass against those hand-drawn reference boxes. It ran the detector only; no face recognition or embeddings were used. The detector selected its largest face proposal in each image. The 240 annotations are balanced across four test sequences and three eye-distance strata (20 frames per stratum in each sequence).

## Results

The automatic proposals were detected on 240/240 frames. Mean detector confidence was 0.857; median IoU with that earlier reference set was 0.624; median center distance was 13.33 px; median proposal-to-reference area ratio was 0.646. The earlier boxes exactly matched the corresponding landmark-proxy coordinates (IoU 1.000 for all 240 frames). These figures describe only the superseded reference set and must not be interpreted as results for the latest hand-drawn boxes. This proposal pass is separate from the four fine-tuned detectors in the main benchmark.

## Files

- `../annotations/manual_boxes_240.csv` — superseded earlier reference coordinates, keyed by sequence and frame number.
- `create_sample.py` — creates a candidate list from the landmark-proxy manifest; it does not overwrite the manual annotation file.
- `run_automated_audit.py` — SCRFD proposals and compact comparison summaries.
- `model_output/summary_by_sequence_and_size.csv` — aggregate proposal comparison by sequence and face-size stratum.
- `model_output/run_metadata.json` — model, runtime, and annotation-source metadata.
- Local per-frame detector outputs and visual overlays are excluded from version control because they contain absolute image paths or face frames.

The pretrained InsightFace weights are licensed for non-commercial research. See the [official SCRFD results](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/README.md) and [model zoo terms](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md).

## Independent-architecture comparison

A second automatic proposal pass used OpenCV Zoo YuNet, which is architecturally distinct from the benchmark detectors. Its statistics also use the superseded reference set, not `boxes_only_240.csv`; see [`../automated_audit_yunet/`](../automated_audit_yunet/). Both model passes remain historical proposal audits and are not the paper's latest manual-box evaluation.

For the 240 audit frames, the target's `person_role_label` is set to `attacker` under the user's assumption; neither face detector infers that role.
