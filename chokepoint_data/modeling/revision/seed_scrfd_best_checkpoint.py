from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed MMDetection's resume metadata with the best completed dev checkpoint.")
    parser.add_argument("--work-dir", default="chokepoint_data/modeling/final_revision/checkpoints/scrfd")
    parser.add_argument("--best-epoch", type=int, required=True)
    parser.add_argument("--best-score", type=float, required=True)
    parser.add_argument("--evidence-log", required=True)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    latest = work_dir / "latest.pth"
    best_epoch = work_dir / f"epoch_{args.best_epoch}.pth"
    if not latest.is_file() or not best_epoch.is_file():
        raise FileNotFoundError(f"Missing resume/best checkpoint: {latest}, {best_epoch}")
    evidence = Path(args.evidence_log).resolve()
    if not evidence.is_file() or f"{args.best_score:.3f}" not in evidence.read_text(encoding="utf-8", errors="replace"):
        raise ValueError("The supplied dev log does not contain the specified best score")

    checkpoint = torch.load(str(latest), map_location="cpu")
    epoch = int(checkpoint.get("meta", {}).get("epoch", 0))
    if epoch <= args.best_epoch:
        raise ValueError(f"Resume epoch {epoch} must be later than seeded best epoch {args.best_epoch}")
    before_sha = sha256_file(latest)
    hook_msgs = checkpoint.setdefault("meta", {}).setdefault("hook_msgs", {})
    hook_msgs["best_score"] = args.best_score
    hook_msgs["best_ckpt"] = str(best_epoch.resolve())
    temp = latest.with_name("latest.pth.seed.tmp")
    torch.save(checkpoint, str(temp))
    temp.replace(latest)
    best_pointer = work_dir / "best_mAP.pth"
    shutil.copy2(best_epoch, best_pointer)

    record = {
        "seeded_utc": datetime.now(timezone.utc).isoformat(),
        "resume_checkpoint": str(latest),
        "resume_epoch": epoch,
        "resume_sha256_before": before_sha,
        "resume_sha256_after": sha256_file(latest),
        "seeded_best_epoch": args.best_epoch,
        "seeded_best_score": args.best_score,
        "seeded_best_checkpoint": str(best_epoch),
        "seeded_best_checkpoint_sha256": sha256_file(best_epoch),
        "best_checkpoint_copy": str(best_pointer),
        "evidence_log": str(evidence),
        "evidence_log_sha256": sha256_file(evidence),
        "weights_and_optimizer_state_modified": False,
    }
    output = Path("chokepoint_data/modeling/final_revision/logs/scrfd_best_seed.json")
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
