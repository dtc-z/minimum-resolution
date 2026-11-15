"""Build a template-styled, deployment-focused revision from the canonical summary."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.shared import Inches
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final"
SOURCE_DOCX = OUT / "Paper_Improved_Final.docx"
OUTPUT_DOCX = OUT / "Paper_Rewritten_Final.docx"
SUMMARY = OUT / "benchmark" / "summary.csv"
FIG_DIR = OUT / "figures" / "rewritten"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "YOLOv5nu-FT": "#0072B2",
    "YOLOv8n-FT": "#009E73",
    "YOLOv12n-FT": "#D55E00",
    "SCRFD-500M-FT": "#CC79A7",
}
SHORT = {
    "YOLOv5nu-FT": "YOLOv5nu-FT",
    "YOLOv8n-FT": "YOLOv8n-FT",
    "YOLOv12n-FT": "YOLOv12n-FT",
    "SCRFD-500M-FT": "SCRFD-500M-FT",
}
MARKERS = {
    "YOLOv5nu-FT": "o",
    "YOLOv8n-FT": "s",
    "YOLOv12n-FT": "^",
    "SCRFD-500M-FT": "D",
}
LINE_STYLES = {
    "YOLOv5nu-FT": "-",
    "YOLOv8n-FT": "--",
    "YOLOv12n-FT": "-.",
    "SCRFD-500M-FT": ":",
}
MODEL_ORDER = list(COLORS)


def load_rows():
    with SUMMARY.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("resolution_width", "resolution_height"):
            row[key] = int(row[key])
        for key in ("conf_threshold", "ap50_vs_derived_boxes", "precision", "recall", "f1",
                    "fps_median", "fps_q1", "fps_q3"):
            row[key] = float(row[key])
    return rows


ROWS = load_rows()
WIDTHS = [640, 480, 384, 320, 288, 256, 240, 192, 160, 128]
THRESHOLDS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.70]


def record(model, width, threshold=0.25):
    return next(r for r in ROWS if r["model"] == model and r["resolution_width"] == width
                and abs(r["conf_threshold"] - threshold) < 1e-9)


def style_axes(ax):
    ax.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_accuracy():
    widths = sorted(WIDTHS)
    fig, (ax_ap, ax_recall, ax_fp) = plt.subplots(
        3, 1, figsize=(3.35, 4.85), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1.05]})
    for model in MODEL_ORDER:
        color = COLORS[model]
        rs = [record(model, w) for w in widths]
        xs = [r["resolution_width"] for r in rs]
        common = {"marker": MARKERS[model], "linestyle": LINE_STYLES[model],
                  "ms": 3.1, "lw": 1.2, "color": color,
                  "markeredgecolor": "#202020", "markeredgewidth": 0.25}
        ax_ap.plot(xs, [r["ap50_vs_derived_boxes"] for r in rs], label=SHORT[model], **common)
        ax_recall.plot(xs, [r["recall"] for r in rs], label=SHORT[model], **common)
        fp_per_100 = [100 * float(r["fp"]) / float(r["gt_count"]) for r in rs]
        ax_fp.plot(xs, fp_per_100, label=SHORT[model], **common)
    for ax in (ax_ap, ax_recall):
        ax.axhline(0.95, color="#555555", linestyle="--", lw=0.8)
        ax.set_ylim(0, 1.04)
        style_axes(ax)
        ax.tick_params(labelsize=6.5, pad=1)
    style_axes(ax_fp)
    ax_fp.tick_params(labelsize=6.5, pad=1)
    ax_fp.set_ylim(bottom=0)
    ax_ap.set_ylabel("AP50", fontsize=7.5, labelpad=2)
    ax_recall.set_ylabel("Recall", fontsize=7.5, labelpad=2)
    ax_fp.set_ylabel("FP / 100 frames", fontsize=7.5, labelpad=2)
    ax_fp.set_xlabel("Input frame width (pixels)", fontsize=7.5, labelpad=2)
    for ax in (ax_ap, ax_recall, ax_fp):
        ax.set_xticks(widths)
        show_ticks = {128, 160, 192, 240, 320, 384, 480, 640}
        ax.set_xticklabels([str(w) if w in show_ticks else "" for w in widths])
        ax.set_xlim(110, 660)
        ax.tick_params(axis="x", labelrotation=40, labelsize=6, pad=1)
    ax_ap.set_title("Test accuracy", loc="left", fontsize=7.5, pad=1)
    ax_recall.set_title("Recall", loc="left", fontsize=7.5, pad=1)
    ax_fp.set_title("False alarms", loc="left", fontsize=7.5, pad=1)

    manifest = ROOT / "chokepoint_data" / "modeling" / "final_revision" / "manifests" / "test.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        median_box_width = float(np.median([float(r["box_w"]) for r in csv.DictReader(f)]))
    scale = median_box_width / 800.0
    top = ax_ap.secondary_xaxis("top", functions=(lambda x: x * scale, lambda x: x / scale))
    top.set_xticks(np.array([128, 240, 320, 480, 640]) * scale)
    top.set_xticklabels([f"{x * scale:.0f}" for x in [128, 240, 320, 480, 640]])
    top.set_xlabel("Median projected proxy-box width (pixels)", fontsize=6.5, labelpad=1)
    top.tick_params(labelsize=5.8, pad=1)
    handles, labels = ax_ap.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, frameon=False, fontsize=6.2, columnspacing=0.7, handlelength=1.2)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.91, bottom=0.18, hspace=0.52)
    save_figure(fig, "fig01_accuracy_by_resolution")


def draw_throughput():
    widths = sorted(WIDTHS)
    fig, (ax, ax_tradeoff) = plt.subplots(
        2, 1, figsize=(3.35, 4.5), gridspec_kw={"height_ratios": [1.1, 1]})
    for model in MODEL_ORDER:
        rs = [record(model, w) for w in widths]
        xs = np.array([r["resolution_width"] for r in rs])
        med = np.array([r["fps_median"] for r in rs])
        low = med - np.array([r["fps_q1"] for r in rs])
        high = np.array([r["fps_q3"] for r in rs]) - med
        ax.errorbar(xs, med, yerr=np.vstack([low, high]), marker=MARKERS[model],
                    linestyle=LINE_STYLES[model], ms=3.1, lw=1.2,
                    markeredgecolor="#202020", markeredgewidth=0.25,
                    capsize=1.5, elinewidth=0.65, color=COLORS[model], label=SHORT[model])
        ap = np.array([r["ap50_vs_derived_boxes"] for r in rs])
        ax_tradeoff.plot(ap, med, marker=MARKERS[model], linestyle=LINE_STYLES[model],
                         ms=3.0, lw=1.0, markeredgecolor="#202020", markeredgewidth=0.25,
                         color=COLORS[model], label=SHORT[model])
        eligible = [r for r in rs if r["ap50_vs_derived_boxes"] >= 0.95 and r["recall"] >= 0.95]
        if eligible:
            candidate = min(eligible, key=lambda r: r["resolution_width"])
            ax_tradeoff.scatter(candidate["ap50_vs_derived_boxes"], candidate["fps_median"],
                                marker=MARKERS[model], s=34, facecolor="white",
                                edgecolor="#111111", linewidth=0.65, zorder=5)
    style_axes(ax)
    ax.set_ylabel("Median FPS (IQR)", fontsize=7.2, labelpad=2)
    ax.set_title("Throughput by input width", loc="left", fontsize=7.5, pad=1)
    ax.set_xticks(widths)
    show_ticks = {128, 160, 192, 240, 320, 384, 480, 640}
    ax.set_xticklabels([str(w) if w in show_ticks else "" for w in widths])
    ax.set_xlim(110, 660)
    ax.tick_params(axis="x", labelrotation=40, labelsize=6, pad=1)
    ax.tick_params(axis="y", labelsize=6.3, pad=1)
    ax.legend(loc="best", frameon=False, fontsize=5.8, ncol=2,
              columnspacing=0.6, handlelength=1.1, borderpad=0.1)
    style_axes(ax_tradeoff)
    ax_tradeoff.axvline(0.95, color="#555555", linestyle="--", lw=0.7)
    ax_tradeoff.set_xlim(0, 1.03)
    ax_tradeoff.set_ylim(35, 80)
    ax_tradeoff.set_xlabel("Test AP50 (score floor 0.001)", fontsize=7.2, labelpad=2)
    ax_tradeoff.set_ylabel("Median FPS", fontsize=7.2, labelpad=2)
    ax_tradeoff.set_title("Accuracy-throughput profile", loc="left", fontsize=7.5, pad=1)
    ax_tradeoff.tick_params(labelsize=6.2, pad=1)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.97, bottom=0.13, hspace=0.5)
    save_figure(fig, "fig02_throughput_by_resolution")


def draw_threshold_sensitivity():
    fig, (ax_recall, ax_f1) = plt.subplots(2, 1, figsize=(3.35, 3.45), sharex=True)
    for model in MODEL_ORDER:
        rs = [record(model, 288, t) for t in THRESHOLDS]
        xs = [r["conf_threshold"] for r in rs]
        common = {"marker": MARKERS[model], "linestyle": LINE_STYLES[model],
                  "ms": 3.1, "lw": 1.2, "markeredgecolor": "#202020",
                  "markeredgewidth": 0.25, "color": COLORS[model], "label": SHORT[model]}
        ax_recall.plot(xs, [r["recall"] for r in rs], **common)
        ax_f1.plot(xs, [r["f1"] for r in rs], **common)
    for ax in (ax_recall, ax_f1):
        ax.axvline(0.25, color="#444444", linestyle="--", lw=0.8)
        ax.axhline(0.95, color="#777777", linestyle=":", lw=0.7)
        ax.set_ylim(0, 1.04)
        style_axes(ax)
        ax.tick_params(labelsize=6.5, pad=1)
    ax_recall.set_title("Threshold sensitivity at 288 px", loc="left", fontsize=7.5, pad=1)
    ax_recall.set_ylabel("Recall", fontsize=7.2, labelpad=2)
    ax_f1.set_ylabel("F1", fontsize=7.2, labelpad=2)
    ax_f1.set_xlabel("Confidence threshold", fontsize=7.2, labelpad=2)
    ax_f1.set_xticks(THRESHOLDS)
    ax_f1.set_xticklabels([f"{x:.2f}" for x in THRESHOLDS])
    handles, labels = ax_recall.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, frameon=False, fontsize=5.8, columnspacing=0.7, handlelength=1.1)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.94, bottom=0.18, hspace=0.35)
    save_figure(fig, "fig03_threshold_sensitivity")


def draw_f1_sensitivity_by_model():
    widths = sorted(WIDTHS)
    fig, axes = plt.subplots(4, 1, figsize=(3.35, 5.2), sharex=True)
    image = None
    for ax, model in zip(axes, MODEL_ORDER):
        matrix = np.array([[record(model, width, threshold)["f1"] for width in widths]
                           for threshold in THRESHOLDS])
        image = ax.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest",
                           cmap="cividis", vmin=0, vmax=1)
        ax.set_title(SHORT[model], loc="left", fontsize=7.2, pad=1)
        ax.set_yticks(range(len(THRESHOLDS)), [f"{t:.2f}" for t in THRESHOLDS])
        ax.set_ylabel("Conf.", fontsize=6.2, labelpad=1)
        ax.tick_params(axis="y", labelsize=5.8, pad=1)
        ax.tick_params(axis="x", labelsize=5.8, pad=1)
    tick_widths = [128, 192, 240, 288, 320, 384, 480, 640]
    axes[-1].set_xticks([widths.index(w) for w in tick_widths], [str(w) for w in tick_widths])
    axes[-1].set_xlabel("Input frame width (pixels)", fontsize=6.8, labelpad=1)
    fig.subplots_adjust(left=0.18, right=0.81, top=0.98, bottom=0.09, hspace=0.33)
    cax = fig.add_axes([0.86, 0.12, 0.025, 0.78])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("F1", fontsize=6.5, labelpad=2)
    colorbar.ax.tick_params(labelsize=5.8, pad=1)
    save_figure(fig, "fig04_f1_sensitivity_by_model")


def replace_paragraph(paragraph, text):
    runs = paragraph.runs
    if runs:
        runs[0].text = text
        for run in runs[1:]:
            run._element.getparent().remove(run._element)
    else:
        paragraph.add_run(text)


def set_cell(cell, text):
    if cell.paragraphs:
        replace_paragraph(cell.paragraphs[0], text)
        for p in cell.paragraphs[1:]:
            replace_paragraph(p, "")
    else:
        cell.add_paragraph(text)


def replace_picture(paragraph, image_path, width_inches=3.15):
    p = paragraph._p
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    paragraph.add_run().add_picture(str(image_path), width=Inches(width_inches))


def append_reference(doc, text, style):
    p = doc.add_paragraph(style=style)
    p.add_run(text)


def build_docx():
    shutil.copy2(SOURCE_DOCX, OUTPUT_DOCX)
    doc = Document(OUTPUT_DOCX)
    p = doc.paragraphs
    replacements = {
        0: "Practical Resolution Selection for Surveillance Face Detection: An Accuracy-Throughput Benchmark of YOLO and SCRFD",
        2: ("Abstract—This study provides a deployment-oriented benchmark for selecting face-detector input resolution under an explicit accuracy and throughput budget. Four fine-tuned models (YOLOv5nu, YOLOv8n, YOLOv12n, and SCRFD-500M) are evaluated on sequence-disjoint ChokePoint Portal 1 indoor frontal video: 4,900 training, 1,135 development, and 5,676 test frames. The test sweep covers ten input widths (640–128 pixels) and six confidence thresholds. A development-only rule selected YOLOv8n-FT at 288 pixels; on the locked test set it reached AP50 1.000, recall 0.998, and median detector throughput 72.1 frames/s on an NVIDIA GeForce RTX 3050 Laptop GPU. At the preset AP50 and recall gate of 0.95, the smallest widths observed in the test sweep were 240 pixels for YOLOv5nu-FT and YOLOv8n-FT, 288 for YOLOv12n-FT, and 384 for SCRFD-500M-FT. These profiles show where resolution reduction preserves proxy-label agreement and where its speed benefit is model-dependent. The benchmark supports model and resolution shortlisting for this surveillance setting; its labels are eye-landmark-derived boxes, and it does not measure recognition accuracy or cross-hardware performance."),
        3: "Index Terms—face detection, surveillance video, resolution scaling, throughput, operating point, YOLO, SCRFD.",
        4: "I. INTRODUCTION",
        5: ("Surveillance pipelines often process every frame through detection before face alignment and identity recognition. Input downscaling can reduce the detector workload, but it also removes spatial detail from already small faces. For deployment, the useful question is therefore not which detector has the highest score at one resolution, but which model, resolution, and confidence setting meets a stated detection-quality requirement within the available processing budget."),
        6: ("This paper provides a practical reference for that decision on ChokePoint Portal 1 indoor frontal video. Four face detectors are fine-tuned on the same sequence-level split and measured across ten input widths and six confidence thresholds. We report detection quality and synchronized throughput together, include a development-locked operating point, and summarize the lowest test-sweep widths that meet the study's preset AP50 and recall gate. The resulting profiles help practitioners shortlist configurations by accuracy and frame-rate priorities. Throughput is measured on one specified laptop GPU; the scores use transparent eye-landmark-derived proxy boxes."),
        7: "II. METHODOLOGY",
        8: "A. Related Work",
        9: ("Low-resolution face research identifies image detail as a key constraint on face analysis [2]. Viola and Jones established an early cascade approach to fast face detection [1], while YOLO uses a single-stage formulation [3]. The YOLOv5 and YOLOv8 checkpoints used here are documented implementations [4], [5], [10]. YOLOv12 uses an attention-centric design [6]. SCRFD is a face-specific detector designed around sample and computation redistribution [7]. ChokePoint provides surveillance video sequences and eye landmarks for face-analysis research [8], [9]."),
        10: "B. Dataset and Sequence Split",
        11: ("The source is the full-frame Portal 1 indoor frontal subset of ChokePoint. Complete source sequences, rather than random frames, define the split: three sequences for training, one for development, and four for the locked test set. No sequence, dataset/frame pair, or adjacent-frame pair crosses the split. Session S2 and camera C2 occur in training and development; camera C1 occurs in training and test, while source sequences and frames remain disjoint. The split does not claim identity-disjoint subjects."),
        12: "TABLE I. Sequence-level data partition.",
        14: ("The split contains 4,900 training, 1,135 development, and 5,676 test frames. Test data were excluded from fitting, checkpoint selection, and operating-point selection. These counts describe labeled full-frame instances in the derived manifest, not every frame or identity in ChokePoint."),
        15: "C. Target Boxes and Annotation Scope",
        16: ("ChokePoint supplies manually labeled eye landmarks, not face rectangles. For inter-eye distance d and eye midpoint (cx, cy), each reference box is defined as x1=cx−1.4d, x2=cx+1.4d, y1=cy−1.2d, and y2=cy+2.4d, then clipped to the 800×600 frame. These fixed geometric boxes are proxy labels. AP50, precision, recall, and F1 in the full sweep measure agreement with that box construction rather than with independently drawn face boundaries."),
        17: ("A separate user-designated reference file covers 240 test frames, balanced across the four test sequences and three eye-distance strata. It is reported only as a secondary box-agreement comparison and was not used to choose a model, resolution, or threshold."),
        18: ("The 240 reference-box coordinates exactly match the accompanying InsightFace proposal file. Consequently, this subset does not provide independent human-annotation validation of those proposals; no second-annotator agreement or adjudication is available. The paper reports the comparison as a supplementary agreement result and keeps the full-sweep metric definition tied to the eye-landmark-derived proxies."),
        19: "D. Detectors and Fine-Tuning",
        20: ("The compared detectors are YOLOv5nu, YOLOv8n, and YOLOv12n initialized from Ultralytics nano weights, and SCRFD-500M initialized from the InsightFace model zoo [7], [10], [11]. All four models use the same 4,900 training instances and 1,135 development instances. YOLO models were trained for 20 epochs at 640-pixel width, batch size 8, seed 391, using Ultralytics 8.4.54 and its recorded automatic optimizer selection. SCRFD was trained for 20 epochs, batch size 2, seed 391, with the recorded MMDetection recipe (SGD, learning rate 0.0025). Framework-specific optimizer and augmentation recipes differ; data partitions and selection data are shared."),
        21: ("The SCRFD code revision, Windows compatibility patch, checkpoint hashes, and training provenance are recorded in the run artifacts. The measured environment used PyTorch 2.1.2 with CUDA 12.1, Ultralytics 8.4.54, MMCV 1.7.2, MMDetection 2.7.0, and OpenCV 4.11.0. The InsightFace pretrained weights used for initialization are designated for non-commercial research; the paper does not imply commercial-use rights [11]."),
        22: "E. Metrics and Measurement Protocol",
        23: ("The aspect ratio is preserved at input widths 640, 480, 384, 320, 288, 256, 240, 192, 160, and 128 pixels. Confidence thresholds are 0.25, 0.35, 0.45, 0.55, 0.65, and 0.70; matching uses IoU 0.50. AP50 is computed from detections above a 0.001 confidence floor with 101-point interpolated precision. At each listed confidence threshold, predictions are matched to unmatched proxy boxes within the same frame; unmatched detections and boxes contribute to false positives and false negatives. The predeclared development rule requires AP50≥0.95 and recall≥0.95 at confidence 0.25, then selects the smallest eligible width. Ties are resolved by development AP50, recall, and a fixed model order. Test results do not change that selection."),
        24: ("Throughput uses a fixed sample of 600 test frames (150 per test sequence), 50 warm-up iterations, and five full repeats, with GPU synchronization around each timed repeat. The timed region includes image reading, framework preprocessing and resizing, detector inference, and model postprocessing; model loading and metric computation are excluded. Median frames/s and the interquartile range are reported for an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB memory. This is detector throughput, not end-to-end recognition latency."),
        25: "III. RESULTS",
        26: "A. Full-Resolution Reference",
        27: "TABLE II. Test results at 640-pixel width. AP50 uses score floor 0.001; recall uses confidence 0.25.",
        29: ("At 640 pixels all four fine-tuned detectors have AP50 1.000 and recall at or above 0.999 against the proxy boxes. Throughput ranges from 39.7 frames/s for SCRFD-500M-FT to 71.4 frames/s for YOLOv8n-FT. The near-ceiling scores make the full-resolution point a useful reference but do not separate the models; their operating profiles diverge under downscaling."),
        30: "B. Resolution, Confidence, and Throughput Profiles",
        31: ("The development-only rule selected YOLOv8n-FT at 288×216 pixels and confidence 0.25 (development AP50 0.9901; recall 0.9630). On the locked test set this configuration reached AP50 1.000, precision 1.000, recall 0.998, F1 0.999, and median throughput 72.1 frames/s (IQR 72.0–73.5). At 30 frames/s per stream, this is 2.4× detector-stage throughput on the measured GPU; at 60 frames/s it is 1.2× before any downstream recognition or multi-stream overhead."),
        32: ("Fig. 1 separates the accuracy response by model. YOLOv5nu-FT and YOLOv8n-FT retain recall above 0.95 through 240 pixels in the test sweep; YOLOv12n-FT does so through 288 pixels, while SCRFD-500M-FT requires 384 pixels. At 192 pixels and below, recall falls below 0.84 for every model. Fig. 2 shows that resolution reduction does not produce a common speed gain: YOLO throughput stays within a narrow band across widths, whereas SCRFD becomes faster as input width shrinks but loses recall sharply. At 128 pixels, SCRFD reaches 69.1 frames/s with recall 0.026; the additional throughput does not compensate for the detection loss under the preset quality gate."),
        34: "Fig. 1. Test AP50 and recall versus input width. AP50 uses score floor 0.001; recall uses confidence 0.25. The dashed line marks the 0.95 reference gate.",
        36: "Fig. 2. Median detector throughput and interquartile range over five repeats on the fixed 600-frame speed sample.",
        38: "Fig. 3. Test recall and F1 across confidence thresholds at 288-pixel width. Dashed guides mark confidence 0.25 and score 0.95.",
        39: "C. Supplementary 240-Frame Box Comparison",
        40: "TABLE III. Agreement with the user-designated 240-frame reference file at 288-pixel width. AP50 uses confidence floor 0.001; precision, recall, and F1 use 0.25.",
        42: ("At 288 pixels, YOLOv5nu-FT obtained AP50 0.965 and recall 0.967 against the 240-frame reference file; YOLOv8n-FT obtained 0.944 and 0.954, YOLOv12n-FT 0.948 and 0.942, and SCRFD-500M-FT 0.286 and 0.371. Median IoU between the reference boxes and the landmark-derived boxes was 0.625 (IQR 0.593–0.654), with 239/240 pairs at or above 0.50. Because the reference coordinates match the InsightFace proposals exactly, this is a cross-box agreement result, not independent evidence of accuracy against human-drawn face rectangles."),
        43: "D. Model-Specific Test Operating Envelopes",
        44: "TABLE IV. Smallest test-sweep width meeting the preset AP50 and recall gate at confidence 0.25.",
        46: ("The gate values were set before test evaluation, but the widths in Table IV are descriptive minima computed from the completed test sweep. They do not replace the development-locked choice of YOLOv8n-FT at 288 pixels. The table supplies a comparable reference profile for shortlisting detectors; operating-point selection for a new deployment should use its own development data, especially when its camera, face scale, or hardware differs."),
        47: "IV. DISCUSSION AND CONCLUSION",
        48: "A. Deployment Interpretation",
        49: ("The results support several concrete choices for the measured Portal 1 setting. For a balanced default, use the development-selected YOLOv8n-FT at 288 pixels and confidence 0.25: it meets the preset quality gate and measures 72.1 frames/s. If preserving quality is more important than reducing input size, 320–640 pixels keeps YOLOv5nu-FT and YOLOv8n-FT near ceiling, with throughput around 68–76 frames/s on this GPU. For a face-specific SCRFD deployment, 384 pixels is the lowest observed test width that meets both gate values; its 51.0 frames/s is slower than the YOLO profiles here. A 240-pixel YOLOv5nu or YOLOv8n result is a useful lower-width test reference, but it was not selected by the development-only rule. Widths at or below 192 pixels are poor choices for all four models because recall drops below 0.84. Raising YOLOv8n-FT confidence from 0.25 to 0.70 at 288 pixels leaves precision at 1.000 while reducing recall from 0.998 to 0.983; the lower threshold is preferable when missed faces carry greater cost."),
        50: "B. Scope and Technical Limitations",
        51: ("The accuracy sweep measures agreement with eye-landmark-derived proxy boxes, not manually drawn face boundaries. The 240-frame secondary file is not independent of InsightFace proposals, and no independent second annotator or adjudication is available. Evaluation covers Portal 1 indoor frontal sequences, not outdoor scenes, other portals, heavy occlusion, or identity-disjoint subjects. Throughput was measured on one RTX 3050 Laptop GPU under the stated software and preprocessing path; the observed rates must not be treated as measured rates on other GPUs, CPUs, accelerators, precisions, or multi-stream loads. The study also does not measure recognition accuracy, face-count stress, or full-pipeline latency. These boundaries define where the profiles are directly applicable and which technical variables future deployment studies should extend."),
        52: "C. Conclusion",
        53: ("This sequence-disjoint benchmark turns resolution, model, and confidence sweeps into practical detector-selection profiles for ChokePoint Portal 1 indoor frontal video. Fine-tuned YOLOv8n-FT at 288 pixels is the development-locked balanced choice on the measured RTX 3050 Laptop GPU, while the test sweep shows that YOLOv5nu and YOLOv8n preserve the 0.95 proxy-quality gate down to 240 pixels, YOLOv12n to 288, and SCRFD to 384. The main deployment lesson is to select resolution by a stated miss tolerance and measured hardware budget: downscaling can yield little speed gain for YOLO, and aggressive reduction can sharply lower recall for every detector. The reported tables and curves provide a reproducible shortlist for this domain; recognition impact and transfer to other hardware and scenes remain separate technical evaluations."),
        54: "REFERENCES",
    }
    for index, text in replacements.items():
        replace_paragraph(p[index], text)

    # Tables retain the current IEEE template table geometry and typography.
    table_i = doc.tables[1]
    split_rows = [
        ("Split", "Sequence", "Frames"),
        ("Train", "P1E_S1_C1", "1,382"),
        ("Train", "P1L_S1_C1", "1,799"),
        ("Train", "P1L_S2_C2", "1,719"),
        ("Development", "P1E_S2_C2", "1,135"),
        ("Test", "P1E_S3_C3", "929"),
        ("Test", "P1E_S4_C1", "1,531"),
        ("Test", "P1L_S3_C3", "1,205"),
        ("Test", "P1L_S4_C1", "2,011"),
    ]
    for row, values in zip(table_i.rows, split_rows):
        for cell, value in zip(row.cells, values):
            set_cell(cell, value)

    baseline = [
        ("Model", "AP50", "Recall", "FPS"),
        ("YOLOv5nu-FT", "1.000", "1.000", "70.3"),
        ("YOLOv8n-FT", "1.000", "0.999", "71.4"),
        ("YOLOv12n-FT", "1.000", "1.000", "51.7"),
        ("SCRFD-500M-FT", "1.000", "1.000", "39.7"),
    ]
    for row, values in zip(doc.tables[2].rows, baseline):
        for cell, value in zip(row.cells, values):
            set_cell(cell, value)

    audit = [
        ("Model", "AP50", "Precision", "Recall", "F1"),
        ("YOLOv5nu-FT", "0.965", "0.983", "0.967", "0.975"),
        ("YOLOv8n-FT", "0.944", "0.958", "0.954", "0.956"),
        ("YOLOv12n-FT", "0.948", "0.983", "0.942", "0.962"),
        ("SCRFD-500M-FT", "0.286", "0.443", "0.371", "0.404"),
    ]
    for row, values in zip(doc.tables[3].rows, audit):
        for cell, value in zip(row.cells, values):
            set_cell(cell, value)

    profiles = [
        ("Model", "Min. width", "AP50", "Recall", "FPS"),
        ("YOLOv5nu-FT", "240", "0.989", "0.962", "68.4"),
        ("YOLOv8n-FT", "240", "0.990", "0.981", "71.5"),
        ("YOLOv12n-FT", "288", "0.990", "0.967", "52.1"),
        ("SCRFD-500M-FT", "384", "0.980", "0.958", "51.0"),
    ]
    for row, values in zip(doc.tables[4].rows, profiles):
        for cell, value in zip(row.cells, values):
            set_cell(cell, value)

    # Refresh the embedded plots from the single canonical test summary.
    image_paragraphs = [par for par in doc.paragraphs if par._p.xpath(".//wp:inline")]
    figures = [
        FIG_DIR / "fig01_accuracy_by_resolution.png",
        FIG_DIR / "fig02_throughput_by_resolution.png",
        FIG_DIR / "fig03_threshold_sensitivity.png",
    ]
    for par, image_path in zip(image_paragraphs, figures):
        replace_picture(par, image_path)

    # The existing reference paragraphs retain the original IEEE reference style.
    references = [
        '[1] P. Viola and M. J. Jones, “Robust real-time face detection,” Int. J. Comput. Vis., vol. 57, no. 2, pp. 137–154, 2004.',
        '[2] P. Li, L. Prieto, D. Mery, and P. J. Flynn, “On low-resolution face recognition in the wild: Comparisons and new techniques,” arXiv:1805.11529, 2018.',
        '[3] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, real-time object detection,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition, 2016, pp. 779–788.',
        '[4] G. Jocher et al., “ultralytics/yolov5,” GitHub repository and software release, 2020–2022.',
        '[5] D. Reis, J. Kupec, J. Hong, and A. Daoudi, “Real-time flying object detection with YOLOv8,” arXiv:2305.09972, 2023.',
        '[6] Y. Tian, Q. Ye, and D. Doermann, “YOLOv12: Attention-centric real-time object detectors,” in Advances in Neural Information Processing Systems, vol. 38, 2025, doi: 10.52202/085713-2627.',
        '[7] J. Guo, J. Deng, A. Lattas, and S. Zafeiriou, “Sample and computation redistribution for efficient face detection,” in Proc. Int. Conf. Learning Representations, 2022, doi: 10.48550/arXiv.2105.04714.',
        '[8] Y. Wong, S. Chen, S. Mau, C. Sanderson, and B. C. Lovell, “Patch-based probabilistic image quality assessment for face selection and improved video-based face recognition,” in Proc. IEEE Biometrics Workshop, CVPR Workshops, 2011, pp. 81–88, doi: 10.1109/CVPRW.2011.5981881.',
        '[9] Y. Wong, S. Chen, S. Mau, C. Sanderson, and B. C. Lovell, “ChokePoint Dataset,” Zenodo, version 1, Jun. 22, 2017, doi: 10.5281/zenodo.815657.',
        '[10] Ultralytics, “Models supported by Ultralytics.” [Online]. Available: https://docs.ultralytics.com/models. Accessed: Nov. 15, 2026.',
        '[11] InsightFace, “Model Zoo and model usage.” [Online]. Available: https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md. Accessed: Nov. 15, 2026.',
    ]
    ref_pars = [i for i, par in enumerate(doc.paragraphs) if par.style.name == "references"]
    for i, text in zip(ref_pars, references[:len(ref_pars)]):
        replace_paragraph(doc.paragraphs[i], text)
    ref_style = doc.paragraphs[ref_pars[-1]].style
    for text in references[len(ref_pars):]:
        append_reference(doc, text, ref_style)

    doc.core_properties.title = "Practical Resolution Selection for Surveillance Face Detection"
    doc.core_properties.subject = "Accuracy-throughput benchmark for fine-tuned YOLO and SCRFD detectors"
    doc.save(OUTPUT_DOCX)
    write_markdown(OUTPUT_DOCX)


def write_markdown(docx_path):
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(docx_path)
    lines = []
    image_n = 0
    image_files = [
        "fig01_accuracy_by_resolution.png",
        "fig02_throughput_by_resolution.png",
        "fig03_threshold_sensitivity.png",
    ]
    for el in doc._element.body.iterchildren():
        if el.tag == qn("w:p"):
            par = Paragraph(el, doc)
            if par._p.xpath(".//wp:inline"):
                image_n += 1
                lines.append(f"![Figure {image_n}](figures/rewritten/{image_files[image_n - 1]})")
                continue
            text = par.text.strip()
            if not text:
                continue
            if par.style.name == "paper title":
                lines.extend([f"# {text}", ""])
            elif par.style.name.startswith("Heading") or text == "REFERENCES":
                lines.extend([f"## {text}", ""])
            else:
                lines.extend([text, ""])
        elif el.tag == qn("w:tbl"):
            table = Table(el, doc)
            rows = [[c.text.replace("\n", " / ").strip() for c in row.cells] for row in table.rows]
            if rows:
                lines.append("| " + " | ".join(rows[0]) + " |")
                lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
                lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
                lines.append("")
    (OUT / "Paper_Rewritten_Final.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    draw_accuracy()
    draw_throughput()
    draw_threshold_sensitivity()
    build_docx()
    print(OUTPUT_DOCX)
    for path in sorted(FIG_DIR.glob("fig*")):
        print(path)


if __name__ == "__main__":
    from rebuild_from_2024_template import main as template_main
    template_main()
