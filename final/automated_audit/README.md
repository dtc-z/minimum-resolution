# SCRFD proposal run (separate from the user's manual box annotations)

This folder records an automatic SCRFD comparison pass over a deterministic sample of 240 test frames. The user confirms that the 240 boxes were drawn manually in a separate annotation pass; that coordinate file is not among the files in this folder. The SCRFD proposals below are auxiliary outputs and must not be substituted for those human boxes.

## Run details

- Model: InsightFace `buffalo_l` / SCRFD-10GF, detector module only; no recognition or embeddings were run.
- Input: 640?640, detection threshold 0.10. The largest detected face was selected as the target; all detections are retained in the local proposal CSV.
- Runtime: ONNX Runtime fell back to CPU because the CUDA provider could not load `cudnn64_9.dll`. All 240 frames completed on CPU.
- Sample: seed 391; 20 frames for each of four test sequences and three eye-distance strata. Occlusion was assumed absent based on the user's statement and was not independently assessed.
- Results: detections on 240/240 frames; mean confidence 0.857; median IoU with the landmark-derived proxy box 0.624; median center distance 13.33 px; median detector-box/proxy-box area ratio 0.646.

These values compare an automatic detector with the landmark-derived proxy labels. They do not measure agreement with human boxes, manual-box AP50/recall, or label bias. Red boxes in local overlays are landmark proxies; green boxes are automatic detector proposals.

The pretrained InsightFace weights are licensed for non-commercial research. See the [official SCRFD results](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/README.md) and [model zoo terms](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md).

## Contents

- `create_sample.py` ? deterministic sample selection.
- `run_automated_audit.py` ? detector proposals and aggregate summaries.
- `sample.csv` and `model_output/model_proposals.csv` ? per-frame local outputs with image paths; do not publish without sanitizing those paths.
- `model_output/summary_by_sequence_and_size.csv` ? aggregate summary by test sequence and face-size stratum.
- `model_output/run_metadata.json` ? model and runtime details.
- `model_output/overlays/` and `contact_sheets/` ? local visual review aids containing faces; excluded from version control.
