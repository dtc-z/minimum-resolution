from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def allocate_occlusion(group: pd.DataFrame, n: int) -> pd.DataFrame:
    available = {int(key): int(value) for key, value in group.human_occlusion.value_counts().items()}
    if len(group) < n:
        raise ValueError("A sequence/size stratum has fewer candidates than its final quota")
    quota = {key: 1 for key in available}
    remaining = n - len(quota)
    while remaining > 0:
        capacity = {key: available[key] - quota[key] for key in available if available[key] > quota[key]}
        if not capacity:
            break
        total = sum(capacity.values())
        exact = {key: remaining * capacity[key] / total for key in capacity}
        added = {key: min(capacity[key], int(exact[key])) for key in capacity}
        for key, count in added.items():
            quota[key] += count
        remaining -= sum(added.values())
        if remaining:
            ranked = sorted((key for key in capacity if quota[key] < available[key]),
                            key=lambda key: (-(exact[key] - int(exact[key])), key))
            for key in ranked:
                if not remaining:
                    break
                quota[key] += 1
                remaining -= 1
    pieces = []
    for category in sorted(quota):
        pieces.append(group[group.human_occlusion == category].sample(
            n=quota[category], random_state=391 + int(category) + len(group)))
    return pd.concat(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the final 240 manual audit boxes after occlusion screening.")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--screening-annotations", required=True, help="Annotator A occlusion labels for every candidate")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--per-sequence", type=int, default=60)
    parser.add_argument("--per-size-stratum", type=int, default=20)
    args = parser.parse_args()
    pool = pd.read_csv(args.pool)
    labels = pd.read_csv(args.screening_annotations)
    key = ["dataset", "frame_number"]
    if set(key + ["human_occlusion"]) - set(labels.columns):
        raise ValueError("Screening annotations need dataset, frame_number, and human_occlusion")
    if labels.duplicated(key).any():
        raise ValueError("Screening annotations contain duplicate frame keys")
    data = pool.merge(labels[key + ["human_occlusion"]], on=key, how="left", validate="one_to_one")
    if data.human_occlusion.isna().any() or not data.human_occlusion.isin([0, 1, 2, 3]).all():
        raise ValueError("Every candidate needs a valid occlusion label (0=none, 1=mild, 2=moderate, 3=severe)")

    chosen = []
    for sequence_index, (sequence, sequence_rows) in enumerate(data.groupby("dataset", sort=True)):
        if len(sequence_rows) != len(data) / data.dataset.nunique():
            raise ValueError(f"Unequal screening pool size for sequence {sequence}")
        per_stratum = args.per_sequence // sequence_rows.size_stratum.nunique()
        remainder = args.per_sequence % sequence_rows.size_stratum.nunique()
        for size_index, (size_bin, stratum) in enumerate(sequence_rows.groupby("size_stratum", sort=True)):
            quota = per_stratum + int(size_index < remainder)
            selected = allocate_occlusion(stratum, quota)
            selected = selected.sample(frac=1, random_state=args.seed + sequence_index * 100 + size_index)
            chosen.append(selected)
    final = pd.concat(chosen).sort_values(["dataset", "size_stratum", "frame_number"]).reset_index(drop=True)
    if len(final) != args.per_sequence * data.dataset.nunique():
        raise ValueError("Final sample size differs from requested sequence quotas")

    final["double_annotate"] = False
    for sequence_index, (_, rows) in enumerate(final.groupby("dataset", sort=True)):
        selected_indices = rows.sample(n=round(len(rows) * 0.2), random_state=args.seed + 1000 + sequence_index).index
        final.loc[selected_indices, "double_annotate"] = True
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out, index=False)
    print(f"Wrote final audit sample: {len(final)} boxes; double annotation: {int(final.double_annotate.sum())}; output: {out}")


if __name__ == "__main__":
    main()
