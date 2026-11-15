# Reproducibility artifacts

This directory keeps the sequence split audit, final protocol, and portable SCRFD training configuration. The compact final benchmark and paper are in [`final/`](../../../final/).

Large local artifacts are not published here: extracted ChokePoint frames, prepared train/dev image trees, Python environments, trained checkpoints, raw per-detection predictions, and machine-specific logs. The SCRFD config expects the dataset under `dataset_scrfd/` and its initialization checkpoint under `checkpoints/scrfd_500m.pth` when running locally.

The InsightFace source checkout is also omitted because it is a nested Git repository. For SCRFD training, check out upstream InsightFace at commit `c6c3af7bd2d0a0bf4b78c78437357f3521aa5d52` into `source/insightface/`. Upstream repository: https://github.com/deepinsight/insightface

The checked-in protocol and result tables are sufficient to inspect the reported benchmark without the local dataset or checkpoints.
