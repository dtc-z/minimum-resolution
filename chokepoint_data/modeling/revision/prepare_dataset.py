from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from audit_manifest import audit


def box_label(row, width: int, height: int) -> str:
    x1 = max(0.0, min(float(row.x1), width - 1.0))
    y1 = max(0.0, min(float(row.y1), height - 1.0))
    x2 = max(0.0, min(float(row.x2), width - 1.0))
    y2 = max(0.0, min(float(row.y2), height - 1.0))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate box for {row.image_path}")
    xc, yc = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    return f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sequence-disjoint YOLO train/val/test data via hardlinks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dev-sequence", default="P1E_S2_C2")
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-width", type=int, default=800)
    parser.add_argument("--image-height", type=int, default=600)
    parser.add_argument("--materialize", action="store_true", help="Create image hardlinks and label files; abort if hardlinks fail.")
    parser.add_argument("--allow-copy-fallback", action="store_true",
                        help="Copy an image only when Windows refuses to create its hardlink.")
    args = parser.parse_args()

    manifest, out = Path(args.manifest), Path(args.out)
    audit_dir = out / "manifests"
    audit_result = audit(manifest, audit_dir, args.dev_sequence)
    source = pd.read_csv(manifest)
    if "split" not in source.columns:
        source = source.rename(columns={"p1_indoor_split": "split"})
    train_seq = set(audit_result["train_sequences"])
    dev_seq = set(audit_result["dev_sequences"])
    test_seq = set(audit_result["test_sequences"])
    source["final_split"] = source["dataset"].map(
        lambda value: "test" if value in test_seq else ("dev" if value in dev_seq else "train")
    )

    target = out / "dataset_yolo"
    expected = target / "images"
    if args.materialize and target.exists() and any(expected.glob("*/*")):
        raise FileExistsError(f"Refusing to overwrite existing materialized dataset: {target}")

    dataset_summary = {"seed": args.seed, "dev_sequence": args.dev_sequence,
                       "audit_manifest_sha256": audit_result["source_manifest_sha256"],
                       "test_used_for_training_or_selection": False, "materialized": args.materialize,
                       "copy_fallback_enabled": args.allow_copy_fallback,
                       "splits": {}}
    for split in ("train", "dev", "test"):
        subset = source[source.final_split == split].copy()
        csv_path = out / "manifests" / f"{split}.csv"
        subset.to_csv(csv_path, index=False)
        count = {"instances": int(len(subset)), "sequences": sorted(subset.dataset.unique().tolist()),
                 "manifest": str(csv_path)}
        if args.materialize:
            image_dir = target / "images" / ("val" if split == "dev" else split)
            label_dir = target / "labels" / ("val" if split == "dev" else split)
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            modes = {"hardlink": 0, "copy": 0}
            list_paths = []
            for row in subset.itertuples(index=False):
                src = Path(row.image_path)
                name = f"{row.dataset}_{int(row.frame_number):08d}{src.suffix.lower() or '.jpg'}"
                dst = image_dir / name
                label = label_dir / f"{Path(name).stem}.txt"
                if not dst.exists():
                    try:
                        os.link(src, dst)
                    except OSError:
                        if not args.allow_copy_fallback:
                            raise
                        shutil.copy2(src, dst)
                        modes["copy"] += 1
                    else:
                        modes["hardlink"] += 1
                label.write_text(box_label(row, args.image_width, args.image_height), encoding="utf-8")
                list_paths.append(dst.resolve().as_posix())
            (target / f"{split}.txt").write_text("\n".join(list_paths) + "\n", encoding="utf-8")
            count.update(modes)
        dataset_summary["splits"][split] = count

    if args.materialize:
        yaml = (f"path: '{target.resolve().as_posix()}'\n"
                "train: images/train\nval: images/val\ntest: images/test\n"
                "names:\n  0: face\n")
        yaml_path = target / "chokepoint_face.yaml"
        yaml_path.write_text(yaml, encoding="utf-8")
        dataset_summary["yolo_yaml"] = str(yaml_path.resolve())
    (out / "manifests" / "prepared_dataset.json").write_text(
        json.dumps(dataset_summary, indent=2), encoding="utf-8")
    print(json.dumps(dataset_summary, indent=2))


if __name__ == "__main__":
    main()
