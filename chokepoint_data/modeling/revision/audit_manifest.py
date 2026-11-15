from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "dataset", "session", "camera", "frame_number", "image_path",
    "has_image", "x1", "y1", "x2", "y2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def audit(manifest: Path, out: Path, dev_sequence: str, image_root: Path | None = None) -> dict:
    frame = pd.read_csv(manifest)
    if "split" not in frame.columns and "p1_indoor_split" in frame.columns:
        frame = frame.rename(columns={"p1_indoor_split": "split"})
    missing_columns = sorted(REQUIRED - set(frame.columns) | (set() if "split" in frame.columns else {"split"}))
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {missing_columns}")
    if frame["image_path"].isna().any() or frame["dataset"].isna().any():
        raise ValueError("Manifest contains empty image_path or dataset values")

    frame["split"] = frame["split"].astype(str).str.strip()
    if set(frame["split"].unique()) != {"train_dev", "eval"}:
        raise ValueError(f"Expected source splits train_dev/eval; found {sorted(frame['split'].unique())}")

    duplicates_path = frame[frame.duplicated("image_path", keep=False)]
    duplicates_frame = frame[frame.duplicated(["dataset", "frame_number"], keep=False)]
    sequence_splits = frame.groupby("dataset")["split"].nunique()
    mixed_sequences = sequence_splits[sequence_splits > 1].index.tolist()
    if len(duplicates_path) or len(duplicates_frame) or mixed_sequences:
        raise ValueError(
            "Split audit failed: duplicate image_path=%d, duplicate dataset/frame=%d, mixed sequences=%s"
            % (len(duplicates_path), len(duplicates_frame), mixed_sequences)
        )

    if image_root is None:
        image_exists = frame["image_path"].map(lambda value: Path(str(value)).is_file())
        image_check_mode = "source_image_paths"
    else:
        def materialized_path(row):
            split = "test" if row.split == "eval" else ("dev" if row.dataset == dev_sequence else "train")
            split_dir = "val" if split == "dev" else split
            suffix = Path(str(row.image_path)).suffix.lower() or ".jpg"
            name = f"{row.dataset}_{int(row.frame_number):08d}{suffix}"
            return image_root / "images" / split_dir / name
        image_exists = frame.apply(lambda row: materialized_path(row).is_file(), axis=1)
        image_check_mode = "materialized_dataset_hardlinks"
    claimed_image = truthy(frame["has_image"])
    missing_claimed = frame[claimed_image & ~image_exists]
    image_present_unclaimed = frame[~claimed_image & image_exists]
    if len(missing_claimed):
        raise ValueError(f"{len(missing_claimed)} rows claim has_image=true but the file is missing")

    sequence_split = frame.groupby("dataset")["split"].first().to_dict()
    train_dev_sequences = sorted(key for key, value in sequence_split.items() if value == "train_dev")
    eval_sequences = sorted(key for key, value in sequence_split.items() if value == "eval")
    if dev_sequence not in train_dev_sequences:
        raise ValueError(f"Development sequence {dev_sequence!r} is not in train_dev")

    assigned = frame["dataset"].map(
        lambda sequence: "test" if sequence_split[sequence] == "eval"
        else ("dev" if sequence == dev_sequence else "train")
    )
    frame["final_split"] = assigned
    final_mixed = frame.groupby("dataset")["final_split"].nunique()
    if (final_mixed > 1).any():
        raise ValueError("A sequence appears in multiple final splits")

    # Detect same-sequence neighboring frames crossing a final split boundary.
    frame_nums = frame[["dataset", "frame_number", "final_split"]].drop_duplicates()
    frame_nums["frame_number"] = pd.to_numeric(frame_nums["frame_number"], errors="raise").astype(int)
    frame_nums = frame_nums.sort_values(["dataset", "frame_number"])
    adjacent_crossings = []
    for sequence, group in frame_nums.groupby("dataset"):
        rows = list(group[["frame_number", "final_split"]].itertuples(index=False, name=None))
        for (number_a, split_a), (number_b, split_b) in zip(rows, rows[1:]):
            if number_b - number_a <= 1 and split_a != split_b:
                adjacent_crossings.append({"dataset": sequence, "frame_a": int(number_a), "split_a": split_a,
                                           "frame_b": int(number_b), "split_b": split_b})
    if adjacent_crossings:
        raise ValueError(f"Adjacent frames cross splits: {adjacent_crossings[:10]}")

    final_counts = frame.groupby("final_split").agg(instances=("dataset", "size"), sequences=("dataset", "nunique"))
    sequence_counts = frame.groupby(["final_split", "dataset"]).size().rename("instances").reset_index()
    session_camera = frame.groupby(["final_split", "session", "camera"], dropna=False).size().rename("instances").reset_index()

    out.mkdir(parents=True, exist_ok=True)
    sequence_counts.to_csv(out / "sequence_counts.csv", index=False)
    session_camera.to_csv(out / "session_camera_audit.csv", index=False)
    split_rows = []
    for split in ("train", "dev", "test"):
        subset = frame[frame["final_split"] == split]
        split_rows.append({"split": split, "instances": int(len(subset)),
                           "sequences": int(subset["dataset"].nunique()),
                           "missing_images": int((~image_exists[subset.index]).sum())})
    pd.DataFrame(split_rows).to_csv(out / "split_audit.csv", index=False)

    group_overlaps = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        group_overlaps[f"{left}_vs_{right}"] = {}
        for field in ("session", "camera"):
            left_ids = set(frame.loc[frame.final_split == left, field].dropna().astype(str))
            right_ids = set(frame.loc[frame.final_split == right, field].dropna().astype(str))
            group_overlaps[f"{left}_vs_{right}"][field] = sorted(left_ids & right_ids)
    result = {
        "status": "passed",
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": sha256(manifest),
        "rows": int(len(frame)),
        "dev_sequence": dev_sequence,
        "train_sequences": sorted(frame.loc[frame.final_split == "train", "dataset"].unique().tolist()),
        "dev_sequences": sorted(frame.loc[frame.final_split == "dev", "dataset"].unique().tolist()),
        "test_sequences": sorted(frame.loc[frame.final_split == "test", "dataset"].unique().tolist()),
        "counts": {key: {"instances": int(row.instances), "sequences": int(row.sequences)}
                   for key, row in final_counts.iterrows()},
        "duplicate_image_path_rows": int(len(duplicates_path)),
        "duplicate_dataset_frame_rows": int(len(duplicates_frame)),
        "mixed_source_sequences": mixed_sequences,
        "missing_images_claimed_present": int(len(missing_claimed)),
        "image_check_mode": image_check_mode,
        "files_present_but_has_image_false": int(len(image_present_unclaimed)),
        "adjacent_cross_split_frame_pairs": adjacent_crossings,
        "session_camera_identifier_overlaps": group_overlaps,
        "interpretation": "Session/camera identifiers are reported, not treated as leakage by themselves; sequence and frame identity are the grouping/audit units.",
        "test_policy": "All source eval sequences map to test and are excluded from train/dev and checkpoint selection.",
    }
    (out / "split_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and lock the sequence-level train/dev/test assignment.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dev-sequence", default="P1E_S2_C2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-root", help="Check generated images under this dataset root instead of source paths.")
    args = parser.parse_args()
    result = audit(Path(args.manifest), Path(args.out), args.dev_sequence,
                   Path(args.image_root) if args.image_root else None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
