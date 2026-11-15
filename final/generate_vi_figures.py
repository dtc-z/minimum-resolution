"""Regenerate Vietnamese figures from the packaged canonical benchmark CSV."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY = pd.read_csv(ROOT / "benchmark" / "summary.csv")
TEST = SUMMARY[SUMMARY["split"] == "test"].copy()
MODELS = ["YOLOv5nu-FT", "YOLOv8n-FT", "YOLOv12n-FT", "SCRFD-500M-FT"]
COLORS = {"YOLOv5nu-FT": "#0072B2", "YOLOv8n-FT": "#009E73", "YOLOv12n-FT": "#D55E00", "SCRFD-500M-FT": "#CC79A7"}
MARKERS = {"YOLOv5nu-FT": "o", "YOLOv8n-FT": "s", "YOLOv12n-FT": "^", "SCRFD-500M-FT": "D"}
OUT = ROOT / "figures" / "vi"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7})

# Accuracy and recall by input width at the locked confidence threshold.
accuracy = TEST[TEST["conf_threshold"] == 0.25]
display_widths = sorted(w for w in accuracy["resolution_width"].unique() if w != 256)
fig, axes = plt.subplots(2, 1, figsize=(3.35, 5.0), sharex=True)
for model in MODELS:
    group = accuracy[accuracy["model"] == model].sort_values("resolution_width")
    axes[0].plot(group["resolution_width"], group["ap50_vs_derived_boxes"], color=COLORS[model], marker=MARKERS[model], linewidth=1.5, markersize=4, label=model)
    axes[1].plot(group["resolution_width"], group["recall"], color=COLORS[model], marker=MARKERS[model], linewidth=1.5, markersize=4, label=model)
for ax, title in zip(axes, ["AP50 theo hộp suy ra", "Độ bao phủ (recall)"]):
    ax.set_title(title)
    ax.set_xlabel("Độ rộng khung hình đầu vào (pixel)")
    ax.set_ylim(-0.02, 1.03)
    ax.set_xticks(display_widths)
    ax.tick_params(axis="x", labelrotation=90, labelsize=6.5, pad=2)
    ax.grid(True, alpha=0.25, linewidth=0.6)
axes[0].set_ylabel("Điểm số")
for ax in axes:
    ax.axhline(0.95, color="#666666", linestyle="--", linewidth=0.8)
axes[0].legend(loc="lower right", frameon=False, fontsize=6.5)
fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.13, hspace=0.3)
fig.savefig(OUT / "fig01_accuracy_by_resolution_vi.png", dpi=400, bbox_inches="tight")
fig.savefig(OUT / "fig01_accuracy_by_resolution_vi.pdf", bbox_inches="tight")
plt.close(fig)

# Median throughput and interquartile range.
speed = accuracy.drop_duplicates(["model", "resolution_width"])
fig, ax = plt.subplots(figsize=(3.35, 2.6))
for model in MODELS:
    group = speed[speed["model"] == model].sort_values("resolution_width")
    median = group["fps_median"].to_numpy()
    lower = median - group["fps_q1"].to_numpy()
    upper = group["fps_q3"].to_numpy() - median
    ax.errorbar(group["resolution_width"], median, yerr=[lower, upper], label=model, color=COLORS[model], marker=MARKERS[model], linewidth=1.3, markersize=3, capsize=2)
ax.set_xlabel("Độ rộng khung hình đầu vào (pixel)")
ax.set_ylabel("FPS trung vị (IQR)", labelpad=3)
ax.set_title("Thông lượng theo độ rộng đầu vào", fontsize=8, pad=5)
ax.grid(True, alpha=0.25, linewidth=0.6)
ax.legend(loc="best", frameon=False, fontsize=6.5)
fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.2)
fig.savefig(OUT / "fig02_throughput_by_resolution_vi.png", dpi=400, bbox_inches="tight")
fig.savefig(OUT / "fig02_throughput_by_resolution_vi.pdf", bbox_inches="tight")
plt.close(fig)

# Threshold sensitivity for the operating point selected using dev data.
selection = json.loads((ROOT / "benchmark" / "operating_point_dev_locked.json").read_text(encoding="utf-8"))
model = selection["selected_model"]
width = int(selection["selected_width"])
group = TEST[(TEST["model"] == model) & (TEST["resolution_width"] == width)].sort_values("conf_threshold")
fig, ax = plt.subplots(figsize=(3.35, 2.6))
for column, label, color, marker in [
    ("precision", "Precision", "#0072B2", "o"),
    ("recall", "Độ bao phủ (Recall)", "#D55E00", "s"),
    ("f1", "F1", "#009E73", "^"),
]:
    ax.plot(group["conf_threshold"], group[column], marker=marker, label=label, color=color)
ax.axvline(float(selection["confidence"]), color="#666666", linestyle="--", linewidth=0.8, label="Ngưỡng đã khóa từ dev")
ax.set_xlabel("Ngưỡng tin cậy")
ax.set_ylabel("Điểm số")
ax.set_ylim(-0.02, 1.03)
ax.set_title(f"Độ nhạy ngưỡng: {model}, {width} px", fontsize=8, pad=5)
ax.grid(True, alpha=0.25, linewidth=0.6)
ax.legend(loc="best", frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig03_threshold_sensitivity_vi.png", dpi=400, bbox_inches="tight")
fig.savefig(OUT / "fig03_threshold_sensitivity_vi.pdf", bbox_inches="tight")
plt.close(fig)

print(f"Vietnamese figures saved under {OUT}")
