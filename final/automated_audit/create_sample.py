"""Create a deterministic candidate list from the landmark-proxy test manifest."""

from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
POOL = WORKSPACE / "chokepoint_data/modeling/final_revision/manual_box_audit/screening_pool.csv"
OUT = Path(__file__).resolve().parent / "candidate_sample.csv"

pool = pd.read_csv(POOL)
sample = (
    pool.groupby(["dataset", "size_stratum"], group_keys=False)
    .sample(n=20, replace=False, random_state=391)
    .sort_values(["dataset", "size_stratum", "frame_number"])
    .reset_index(drop=True)
)
sample = sample[[
    "dataset", "frame_number", "image_path", "eye_distance_px", "size_stratum",
    "x1", "y1", "x2", "y2",
]].copy()
sample["occlusion_assumption"] = "none_assumed_from_user_statement"
sample["person_role_label"] = "attacker"
sample["person_role_source"] = "user_assumption"
sample["box_source"] = "landmark_proxy_reference"
sample["selection_seed"] = 391
sample.to_csv(OUT, index=False)
print(f"Wrote {len(sample)} model-proposal samples to {OUT}")
