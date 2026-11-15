from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SCRFD labelv2 train/dev/test from the locked shared boxes.")
    parser.add_argument("--root", required=True, help="Existing final_revision/dataset_yolo root with hardlinked images")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    source_root, out = Path(args.root), Path(args.out)
    if out.exists() and any(path.is_file() for path in out.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite existing SCRFD dataset root: {out}")
    records = {}
    for split, manifest_path in (("train", Path(args.train)), ("dev", Path(args.dev)), ("test", Path(args.test))):
        frame = pd.read_csv(manifest_path)
        image_dir = source_root / "images" / ("val" if split == "dev" else split)
        target_images = out / split / "images"
        target_images.mkdir(parents=True, exist_ok=True)
        lines = []
        linked = 0
        for row in frame.itertuples(index=False):
            suffix = Path(str(row.image_path)).suffix.lower() or ".jpg"
            filename = f"{row.dataset}_{int(row.frame_number):08d}{suffix}"
            source = image_dir / filename
            if not source.is_file():
                raise FileNotFoundError(f"Prepared image absent: {source}")
            destination = target_images / filename
            if not destination.exists():
                os.link(source, destination)
            linked += 1
            coords = (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                raise ValueError(f"Invalid box for {filename}: {coords}")
            # RetinaFaceDataset's labelv2 parser expects a fifth ignore flag; 0 keeps this derived box.
            lines.extend([f"# {filename} {args.width} {args.height}",
                          " ".join(f"{value:.4f}" for value in coords) + " 0"])
        ann = out / split / "labelv2.txt"
        ann.write_text("\n".join(lines) + "\n", encoding="utf-8")
        records[split] = {"instances": int(len(frame)), "images_hardlinked": linked,
                          "sequences": sorted(frame.dataset.unique().tolist()),
                          "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                          "annotation_file": str(ann.resolve()), "image_prefix": str(target_images.resolve())}
    if records["train"]["sequences"] == records["test"]["sequences"] or set(records["train"]["sequences"]) & set(records["test"]["sequences"]):
        raise ValueError("Sequence overlap between SCRFD train and test")
    summary = {"format": "InsightFace SCRFD RetinaFaceDataset labelv2", "box_source": "same locked eye-landmark-derived boxes as YOLO",
               "landmarks": "omitted; fifth label field is ignore=0", "test_used_for_training_or_selection": False,
               "splits": records}
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
