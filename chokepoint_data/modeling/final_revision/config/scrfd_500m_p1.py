from pathlib import Path

_REVISION_ROOT = Path(__file__).resolve().parents[1]
_SCRFD_ROOT = _REVISION_ROOT / "source" / "insightface" / "detection" / "scrfd"
_DATA_ROOT = _REVISION_ROOT / "dataset_scrfd"
_CHECKPOINT = _REVISION_ROOT / "checkpoints" / "scrfd_500m.pth"
_COMMON_TEST_CFG = dict(
    nms_pre=-1,
    min_bbox_size=0,
    score_thr=0.001,
    nms=dict(type="nms", iou_threshold=0.5),
    max_per_img=300,
)

_base_ = str(_SCRFD_ROOT / "configs" / "scrfd" / "scrfd_500m.py")

# Keep the proposal head's actual post-processing aligned with the shared
# confidence floor, NMS IoU, and detection cap used by the YOLO benchmarks.
model = dict(
    bbox_head=dict(test_cfg=_COMMON_TEST_CFG),
    test_cfg=_COMMON_TEST_CFG,
)
test_cfg = _COMMON_TEST_CFG

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        ann_file=str(_DATA_ROOT / "train" / "labelv2.txt"),
        img_prefix=str(_DATA_ROOT / "train" / "images"),
    ),
    val=dict(
        ann_file=str(_DATA_ROOT / "dev" / "labelv2.txt"),
        img_prefix=str(_DATA_ROOT / "dev" / "images"),
    ),
    # The training config exposes no test split to fit/validation.
    test=dict(
        ann_file=str(_DATA_ROOT / "dev" / "labelv2.txt"),
        img_prefix=str(_DATA_ROOT / "dev" / "images"),
    ),
)

optimizer = dict(lr=0.0025)
lr_config = dict(policy="step", warmup="linear", warmup_iters=100,
                 warmup_ratio=0.001, step=[14, 18])
runner = dict(type="EpochBasedRunner", max_epochs=20)
total_epochs = 20
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric="mAP", save_best="mAP", rule="greater")
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
load_from = str(_CHECKPOINT)
resume_from = None
workflow = [("train", 1)]

# MMCV serializes module globals into the run config; remove helper objects so
# only actual configuration values are emitted as valid Python.
del Path, _REVISION_ROOT, _SCRFD_ROOT, _DATA_ROOT, _CHECKPOINT, _COMMON_TEST_CFG
