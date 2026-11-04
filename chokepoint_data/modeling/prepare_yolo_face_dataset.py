from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


def normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "split" not in df.columns and "p1_indoor_split" in df.columns:
        df = df.rename(columns={"p1_indoor_split": "split"})
    if "split" not in df.columns:
        raise ValueError("Manifest must contain either 'split' or 'p1_indoor_split'.")
    return df


def safe_image_name(row) -> str:
    return f"{row.dataset}_{int(row.frame_number):08d}.jpg"


def yolo_label(row, image_width: int = 800, image_height: int = 600) -> str:
    x1 = max(0.0, min(float(row.x1), image_width - 1.0))
    y1 = max(0.0, min(float(row.y1), image_height - 1.0))
    x2 = max(0.0, min(float(row.x2), image_width - 1.0))
    y2 = max(0.0, min(float(row.y2), image_height - 1.0))
    xc = ((x1 + x2) / 2.0) / image_width
    yc = ((y1 + y2) / 2.0) / image_height
    bw = max(1.0, x2 - x1) / image_width
    bh = max(1.0, y2 - y1) / image_height
    return f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n"


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def prepare_split(df: pd.DataFrame, split_name: str, output_root: Path) -> dict:
    image_dir = output_root / "images" / split_name
    label_dir = output_root / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rows = 0
    missing = 0
    hardlinks = 0
    copies = 0
    existing = 0
    image_list = []
    for row in df.itertuples(index=False):
        src = Path(row.image_path)
        if not src.exists():
            missing += 1
            continue
        filename = safe_image_name(row)
        dst_image = image_dir / filename
        dst_label = label_dir / filename.replace(".jpg", ".txt")
        mode = link_or_copy(src, dst_image)
        hardlinks += int(mode == "hardlink")
        copies += int(mode == "copy")
        existing += int(mode == "exists")
        dst_label.write_text(yolo_label(row), encoding="utf-8")
        image_list.append(dst_image.resolve())
        rows += 1

    list_path = output_root / f"{split_name}.txt"
    list_path.write_text("\n".join(str(path) for path in image_list) + "\n", encoding="utf-8")
    return {
        "split": split_name,
        "rows": rows,
        "missing_images": missing,
        "hardlinks": hardlinks,
        "copies": copies,
        "existing": existing,
        "image_list": str(list_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="chokepoint_data/modeling/chokepoint_p1_indoor_frontal_manifest.csv")
    parser.add_argument("--output-root", default="chokepoint_data/modeling/yolo_face_dataset")
    parser.add_argument("--train-limit", type=int, default=0, help="0 means use the full train_dev split.")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means use the full eval split.")
    parser.add_argument("--seed", type=int, default=391)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    df = normalize_manifest(pd.read_csv(args.manifest))

    train_df = df[df["split"] == "train_dev"].copy()
    val_df = df[df["split"] == "eval"].copy()
    if args.train_limit and args.train_limit < len(train_df):
        train_df = train_df.sample(args.train_limit, random_state=args.seed)
    if args.val_limit and args.val_limit < len(val_df):
        val_df = val_df.sample(args.val_limit, random_state=args.seed + 1)

    train_summary = prepare_split(train_df, "train", output_root)
    val_summary = prepare_split(val_df, "val", output_root)

    yaml_text = (
        f"path: {output_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: face\n"
    )
    yaml_path = output_root / "chokepoint_face.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    summary = {
        "dataset_root": str(output_root.resolve()),
        "yaml": str(yaml_path.resolve()),
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "train": train_summary,
        "val": val_summary,
    }
    summary_path = output_root / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
