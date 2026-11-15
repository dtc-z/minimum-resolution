from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a seeded test-only manual-box audit sample.")
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--per-sequence", type=int, default=90,
                        help="Candidate pool size per sequence; final 240 are selected after occlusion screening.")
    args = parser.parse_args()

    data = pd.read_csv(args.test_manifest)
    required = {"dataset", "image_path", "frame_number", "eye_distance_px", "x1", "y1", "x2", "y2"}
    if required - set(data.columns):
        raise ValueError(f"Test manifest missing fields: {sorted(required - set(data.columns))}")
    if set(data.dataset.unique()) != {"P1E_S3_C3", "P1E_S4_C1", "P1L_S3_C3", "P1L_S4_C1"}:
        raise ValueError("Manual audit sampling is restricted to the four locked test sequences")

    selected = []
    for seq_index, (sequence, group) in enumerate(data.groupby("dataset", sort=True)):
        group = group.copy()
        group["size_stratum"] = pd.qcut(group.eye_distance_px.rank(method="first"), 3,
                                        labels=["small", "medium", "large"])
        for size_index, (size_bin, stratum) in enumerate(group.groupby("size_stratum", observed=True)):
            n = args.per_sequence // 3 + int(size_index < args.per_sequence % 3)
            if len(stratum) < n:
                raise ValueError(f"Not enough rows for {sequence}/{size_bin}: need {n}, got {len(stratum)}")
            selected.append(stratum.sample(n=n, random_state=args.seed + seq_index * 100 + size_index))
    sample = pd.concat(selected, ignore_index=True).sort_values(["dataset", "size_stratum", "frame_number"])
    sample["double_annotate"] = False
    for column in ("human_occlusion", "annotator", "human_x1", "human_y1", "human_x2", "human_y2", "adjudicated_x1", "adjudicated_y1", "adjudicated_x2", "adjudicated_y2"):
        sample[column] = ""
    columns = ["dataset", "frame_number", "image_path", "eye_distance_px", "size_stratum", "double_annotate",
               "x1", "y1", "x2", "y2", "human_occlusion", "annotator", "human_x1", "human_y1",
               "human_x2", "human_y2", "adjudicated_x1", "adjudicated_y1", "adjudicated_x2", "adjudicated_y2"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample[columns].to_csv(out, index=False)
    print(f"Wrote {len(sample)} screening candidates to {out}; final stratified sample is selected after occlusion is annotated.")


if __name__ == "__main__":
    main()
