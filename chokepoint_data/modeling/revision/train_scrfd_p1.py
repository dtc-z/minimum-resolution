from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune official SCRFD on the locked ChokePoint train/dev split.")
    parser.add_argument("--config", default="chokepoint_data/modeling/final_revision/config/scrfd_500m_p1.py")
    parser.add_argument("--work-dir", default="chokepoint_data/modeling/final_revision/checkpoints/scrfd")
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--resume-from", default=None, help="resume from this checkpoint; defaults to work-dir/latest.pth if present")
    args = parser.parse_args()

    import torch
    from mmcv import Config

    if not torch.cuda.is_available():
        raise RuntimeError("Refusing SCRFD training: CUDA is unavailable")
    config = Path(args.config).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(str(config))
    checkpoint = Path(cfg.load_from).resolve()
    train_ann = Path(cfg.data.train.ann_file).resolve()
    dev_ann = Path(cfg.data.val.ann_file).resolve()
    test_ann = Path(cfg.data.test.ann_file).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SCRFD initialization checkpoint is missing: {checkpoint}")
    if not train_ann.is_file() or not dev_ann.is_file():
        raise FileNotFoundError(f"SCRFD train/dev annotation missing: {train_ann}, {dev_ann}")
    if test_ann != dev_ann:
        raise ValueError("SCRFD config must not expose the locked test split to training or validation")
    if int(cfg.runner.max_epochs) != 20:
        raise ValueError(f"Expected 20 training epochs, got {cfg.runner.max_epochs}")

    resume_from = Path(args.resume_from).resolve() if args.resume_from else work_dir / "latest.pth"
    if resume_from.is_file():
        resume_checkpoint = torch.load(str(resume_from), map_location="cpu")
        resume_epoch = int(resume_checkpoint.get("meta", {}).get("epoch", 0))
        if resume_epoch >= int(cfg.runner.max_epochs):
            raise ValueError(f"Resume checkpoint is already at epoch {resume_epoch}; expected fewer than 20")
    else:
        resume_from = None

    repo = Path("chokepoint_data/modeling/final_revision/source/insightface/detection/scrfd").resolve()
    command = [
        sys.executable,
        str(repo / "tools" / "train.py"),
        str(config),
        "--work-dir", str(work_dir),
        "--seed", str(args.seed),
        "--deterministic",
    ]
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "wrapper_cwd": str(Path.cwd()),
        "training_cwd": str(repo),
        "config": str(config),
        "config_sha256": sha256_file(config),
        "windows_evaluation_patch": str(Path("chokepoint_data/modeling/final_revision/logs/scrfd_windows_eval.patch").resolve()),
        "windows_evaluation_patch_sha256": sha256_file(Path("chokepoint_data/modeling/final_revision/logs/scrfd_windows_eval.patch")),
        "windows_symlink_fallback_patch": str(Path("chokepoint_data/modeling/final_revision/logs/scrfd_windows_symlink_fallback.patch").resolve()),
        "windows_symlink_fallback_patch_sha256": sha256_file(Path("chokepoint_data/modeling/final_revision/logs/scrfd_windows_symlink_fallback.patch")),
        "mmcv_path_implementation": str(Path("chokepoint_data/modeling/final_revision/envs/scrfd/lib/site-packages/mmcv/utils/path.py").resolve()),
        "mmcv_path_implementation_sha256": sha256_file(Path("chokepoint_data/modeling/final_revision/envs/scrfd/lib/site-packages/mmcv/utils/path.py")),
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "resume_from": str(resume_from) if resume_from else None,
        "resume_checkpoint_sha256": sha256_file(resume_from) if resume_from else None,
        "best_checkpoint_seed_record": str(Path("chokepoint_data/modeling/final_revision/logs/scrfd_best_seed.json").resolve()) if Path("chokepoint_data/modeling/final_revision/logs/scrfd_best_seed.json").is_file() else None,
        "best_checkpoint_seed_record_sha256": sha256_file(Path("chokepoint_data/modeling/final_revision/logs/scrfd_best_seed.json")) if Path("chokepoint_data/modeling/final_revision/logs/scrfd_best_seed.json").is_file() else None,
        "train_annotation": str(train_ann),
        "dev_annotation": str(dev_ann),
        "test_annotation_in_config": str(test_ann),
        "test_used_for_training_or_validation": False,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "seed": args.seed,
        "status": "running",
    }
    record_path = Path("chokepoint_data/modeling/final_revision/logs/scrfd_training_run.json")
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("SCRFD training command:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)
    best_checkpoint = work_dir / "best_mAP.pth"
    if not best_checkpoint.is_file():
        candidates = sorted(
            work_dir.glob("best_mAP_epoch_*.pth"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )
        if not candidates:
            raise FileNotFoundError(f"SCRFD training finished without a best mAP checkpoint in {work_dir}")
        best_checkpoint = candidates[-1]
    best_data = torch.load(str(best_checkpoint), map_location="cpu")
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["status"] = "complete"
    record["best_checkpoint"] = str(best_checkpoint)
    record["best_checkpoint_sha256"] = sha256_file(best_checkpoint)
    record["best_epoch"] = int(best_data.get("meta", {}).get("epoch", 0))
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
