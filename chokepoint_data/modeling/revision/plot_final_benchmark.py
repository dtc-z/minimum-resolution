from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "YOLOv5nu-FT": "#0072B2",
    "YOLOv8n-FT": "#009E73",
    "YOLOv12n-FT": "#D55E00",
    "SCRFD-500M-FT": "#CC79A7",
}
MARKERS = {"YOLOv5nu-FT": "o", "YOLOv8n-FT": "s", "YOLOv12n-FT": "^", "SCRFD-500M-FT": "D"}


def save_figure(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the locked final benchmark summary.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--selection", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    frame = pd.read_csv(summary_path)
    required = {
        "split", "model", "resolution_width", "conf_threshold",
        "ap50_vs_derived_boxes", "recall", "precision", "f1",
        "fps_median", "fps_q1", "fps_q3",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Summary is missing columns: {missing}")
    test = frame[frame["split"] == "test"].copy()
    if test.empty:
        raise ValueError("Summary contains no final test rows")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    operating_conf = 0.25
    accuracy = test[test["conf_threshold"] == operating_conf]
    if accuracy.empty:
        raise ValueError(f"Summary has no rows for confidence threshold {operating_conf}")
    accuracy = accuracy.sort_values("resolution_width")
    # Keep the labels around 240 px readable at column width; the 256 px
    # datapoint remains plotted and its value is listed in the methods.
    display_widths = sorted(w for w in accuracy["resolution_width"].unique() if w != 256)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9,
                         "axes.titlesize": 10, "legend.fontsize": 8})

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 5.0), sharex=True)
    for model, group in accuracy.groupby("model", sort=False):
        color = COLORS.get(model, "#444444")
        marker = MARKERS.get(model, "o")
        axes[0].plot(group["resolution_width"], group["ap50_vs_derived_boxes"],
                     color=color, marker=marker, linewidth=1.5, markersize=4, label=model)
        axes[1].plot(group["resolution_width"], group["recall"],
                     color=color, marker=marker, linewidth=1.5, markersize=4, label=model)
    for ax, title in zip(axes, ["AP50 vs. derived boxes", "Recall"]):
        ax.set_title(title)
        ax.set_xlabel("Input frame width (pixels)")
        ax.set_ylim(-0.02, 1.03)
        ax.set_xticks(display_widths)
        ax.tick_params(axis="x", labelrotation=90, labelsize=6.5, pad=2)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")
        ax.grid(True, alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Score")
    axes[0].axhline(0.95, color="#666666", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.95, color="#666666", linestyle="--", linewidth=0.8)
    axes[0].legend(loc="lower right", frameon=False, fontsize=6.5)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.13, hspace=0.3)
    save_figure(fig, out, "fig01_accuracy_by_resolution")

    speed = accuracy.drop_duplicates(["model", "resolution_width"]).sort_values("resolution_width")
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    for model, group in speed.groupby("model", sort=False):
        color = COLORS.get(model, "#444444")
        marker = MARKERS.get(model, "o")
        median = group["fps_median"].to_numpy()
        lower = median - group["fps_q1"].to_numpy()
        upper = group["fps_q3"].to_numpy() - median
        ax.errorbar(group["resolution_width"], median, yerr=[lower, upper], label=model,
                    color=color, marker=marker, linewidth=1.3, markersize=3, capsize=2)
    ax.set_xlabel("Input frame width (pixels)")
    ax.set_ylabel("Median FPS (IQR)", labelpad=3)
    ax.set_title("Throughput by input width", fontsize=8, pad=5)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="best", frameon=False, fontsize=6.5)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.2)
    save_figure(fig, out, "fig02_throughput_by_resolution")

    selection_path = Path(args.selection) if args.selection else summary_path.parent / "operating_point_dev_locked.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selected_model") is not None:
        selected_model = selection["selected_model"]
        selected_width = int(selection["selected_width"])
        selected = test[(test["model"] == selected_model) & (test["resolution_width"] == selected_width)]
        if selected.empty:
            raise ValueError("Dev-locked operating point is missing from final test summary")
        selected = selected.sort_values("conf_threshold")
        fig, ax = plt.subplots(figsize=(3.35, 2.6))
        ax.plot(selected["conf_threshold"], selected["precision"], marker="o", label="Precision", color="#0072B2")
        ax.plot(selected["conf_threshold"], selected["recall"], marker="s", label="Recall", color="#D55E00")
        ax.plot(selected["conf_threshold"], selected["f1"], marker="^", label="F1", color="#009E73")
        ax.axvline(float(selection["confidence"]), color="#666666", linestyle="--", linewidth=0.8,
                   label="Dev-locked confidence")
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.02, 1.03)
        ax.set_title(f"Threshold sensitivity: {selected_model}, {selected_width} px", fontsize=8, pad=5)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="best", frameon=False, fontsize=7)
        fig.tight_layout()
        save_figure(fig, out, "fig03_threshold_sensitivity")
    else:
        selected = test[test["resolution_width"] == max(test["resolution_width"].unique())].sort_values("conf_threshold")
        fig, ax = plt.subplots(figsize=(3.35, 2.6))
        for model, group in selected.groupby("model", sort=False):
            ax.plot(group["conf_threshold"], group["f1"], marker=MARKERS.get(model, "o"),
                    label=model, color=COLORS.get(model, "#444444"), linewidth=1.3, markersize=3)
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("F1")
        ax.set_ylim(-0.02, 1.03)
        ax.set_title("Threshold sensitivity at 640-pixel width")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="best", frameon=False, fontsize=6.5)
        fig.tight_layout()
        save_figure(fig, out, "fig03_threshold_sensitivity")

    (out / "figure_sources.json").write_text(json.dumps({
        "canonical_summary": str(summary_path.resolve()),
        "canonical_summary_sha256": __import__("hashlib").sha256(summary_path.read_bytes()).hexdigest(),
        "operating_point_file": str(selection_path.resolve()),
        "figures": sorted(path.name for path in out.glob("fig*.pdf")),
        "labels": "eye-landmark-derived proxy boxes; manual audit has to be reported separately",
    }, indent=2), encoding="utf-8")
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
