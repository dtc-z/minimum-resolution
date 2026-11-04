from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT / "chokepoint_data" / "modeling"
FIGURES = MODELING / "ieee_paper_figures_en"
OUTPUT = ROOT / "Paper_IEEE_ChokePoint_Resolution_Benchmark_Revised.docx"
OUTPUT_FALLBACK = ROOT / "Paper_IEEE_ChokePoint_Resolution_Benchmark_Revised_v2.docx"
TEX_OUTPUT = ROOT / "Paper_IEEE_ChokePoint_Resolution_Benchmark_IEEEtran.tex"

FIXED_CSV = MODELING / "benchmark_full_eval_gpu_yolo_both_scrfd_sample5676.csv"
RESOLUTION_CSV = MODELING / "benchmark_resolution_full_eval_gpu_yolov5n_yolov8n_yolov12n_scrfd_sample5676.csv"
MANIFEST_CSV = MODELING / "chokepoint_p1_indoor_frontal_manifest.csv"

TITLE = (
    "Resolution, Accuracy, and Throughput Trade-off for Lightweight Face "
    "Detection in ChokePoint Surveillance Video"
)

MODEL_ORDER = ["YOLOv5n-FT", "YOLOv8n-FT", "YOLOv12n-FT", "SCRFD"]
COLORS = {
    "YOLOv5n-FT": "#2563eb",
    "YOLOv8n-FT": "#16a34a",
    "YOLOv12n-FT": "#dc2626",
    "SCRFD": "#7c3aed",
}
MARKERS = {
    "YOLOv5n-FT": "o",
    "YOLOv8n-FT": "s",
    "YOLOv12n-FT": "^",
    "SCRFD": "D",
}


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def configure_section(section, columns: int = 1) -> None:
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    cols = section._sectPr.xpath("./w:cols")
    if cols:
        cols = cols[0]
    else:
        cols = OxmlElement("w:cols")
        section._sectPr.append(cols)
    cols.set(qn("w:num"), str(columns))
    cols.set(qn("w:space"), "360")


def set_document_defaults(doc: Document) -> None:
    configure_section(doc.sections[0], columns=1)
    styles = doc.styles
    for style_name in ["Normal", "Body Text"]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(10)
    for style_name in ["Heading 1", "Heading 2", "Heading 3"]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.bold = True
        style.font.size = Pt(10)


def set_run_font(run, size: int = 10, bold: bool = False, italic: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_centered(doc: Document, text: str, size: int, bold: bool = False) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)


def clear_table_borders(table) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "nil")


def add_author_block(doc: Document) -> None:
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    clear_table_borders(table)
    authors = [
        (
            "Vu Hai Minh",
            "Dept. of Artificial Intelligence",
            "FPT University",
            "Ha Noi, Viet Nam",
            "vhaiminh.2k5@gmail.com",
        ),
        (
            "Cao Van Mai",
            "Dept. of Artificial Intelligence",
            "FPT University",
            "Ha Noi, Viet Nam",
            "maicv2@fe.edu.vn",
        ),
    ]
    for cell, author in zip(table.rows[0].cells, authors):
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        for idx, line in enumerate(author):
            run = p.add_run(line)
            set_run_font(run, size=10, bold=(idx == 0))
            if idx < len(author) - 1:
                p.add_run("\n")


def add_abstract(doc: Document, abstract: str, keywords: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(0)
    label = p.add_run("Abstract--")
    set_run_font(label, size=9, bold=True, italic=True)
    run = p.add_run(abstract)
    set_run_font(run, size=9, italic=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    label = p.add_run("Index Terms--")
    set_run_font(label, size=9, bold=True, italic=True)
    run = p.add_run(keywords)
    set_run_font(run, size=9, italic=True)


def add_ieee_heading(doc: Document, number: str, title: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(f"{number}. {title.upper()}")
    set_run_font(run, size=10, bold=True)


def add_references_heading(doc: Document) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run("REFERENCES")
    set_run_font(run, size=10, bold=True)


def add_body(doc: Document, text: str, indent: bool = True) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0.16 if indent else 0)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    set_run_font(run, size=10)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style=None)
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.first_line_indent = Inches(-0.12)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run("- " + item)
        set_run_font(run, size=10)


def format_table(table, font_size: int = 7) -> None:
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    set_run_font(run, size=font_size)


def add_table(doc: Document, caption: str, columns: list[str], rows: list[list[str]], wide: bool = False) -> None:
    del wide
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(caption)
    set_run_font(run, size=8, bold=True)
    table = doc.add_table(rows=1, cols=len(columns))
    table.autofit = True
    tbl_w = table._tbl.tblPr.xpath("./w:tblW")
    if tbl_w:
        tbl_w = tbl_w[0]
    else:
        tbl_w = OxmlElement("w:tblW")
        table._tbl.tblPr.insert(0, tbl_w)
    tbl_w.set(qn("w:type"), "pct")
    tbl_w.set(qn("w:w"), "5000")
    for idx, column in enumerate(columns):
        table.rows[0].cells[idx].text = column
    for row in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].text = str(value)
    format_table(table, font_size=6)


def add_figure(doc: Document, filename: str, caption: str, insight: str, wide: bool = True) -> None:
    del wide
    path = FIGURES / filename
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(3.0))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(0)
    r = cap.add_run(caption)
    set_run_font(r, size=8)
    add_body(doc, insight, indent=False)


def add_reference(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(-0.18)
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    set_run_font(run, size=8)


def save_figure(fig, name: str, tight: bool = True) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed = pd.read_csv(FIXED_CSV)
    resolution = pd.read_csv(RESOLUTION_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)
    return fixed, resolution, manifest


def build_cutoff_rows(resolution: pd.DataFrame) -> list[list[str]]:
    primary = resolution[resolution["conf_threshold"].eq(0.25)].copy()
    rows = []
    for model in MODEL_ORDER:
        group = primary[primary["model"].eq(model)].sort_values("resolution_width")
        valid = group[(group["ap50_voc101"] >= 0.95) & (group["recall"] >= 0.95)]
        if valid.empty:
            rows.append([model, "None", "-", "-", "-", "-"])
            continue
        best = valid.iloc[0]
        rows.append([
            model,
            f"{int(best['resolution_width'])}x{int(best['resolution_height'])}",
            best["resolution_label"].split()[0],
            fmt(best["ap50_voc101"]),
            fmt(best["recall"]),
            fmt(best["fps"], 1),
        ])
    return rows


def build_fixed_rows(fixed: pd.DataFrame) -> list[list[str]]:
    rows = []
    keep = ["YOLOv5n", "YOLOv8n", "YOLOv12n", "YOLOv5n-FT", "YOLOv8n-FT", "YOLOv12n-FT", "SCRFD"]
    for _, row in fixed[fixed["model"].isin(keep)].iterrows():
        rows.append([
            row["model"],
            str(row["device"]),
            fmt(row["precision"]),
            fmt(row["recall"]),
            fmt(row["f1"]),
            fmt(row["ap50_voc101"]),
            fmt(row["fps"], 1),
        ])
    return rows


def build_fixed_rows_compact(fixed: pd.DataFrame) -> list[list[str]]:
    rows = []
    keep = ["YOLOv5n", "YOLOv8n", "YOLOv12n", "YOLOv5n-FT", "YOLOv8n-FT", "YOLOv12n-FT", "SCRFD"]
    for _, row in fixed[fixed["model"].isin(keep)].iterrows():
        rows.append([
            row["model"],
            fmt(row["recall"]),
            fmt(row["ap50_voc101"]),
            fmt(row["fps"], 1),
        ])
    return rows


def make_figure_accuracy_resolution(resolution: pd.DataFrame) -> None:
    primary = resolution[resolution["conf_threshold"].eq(0.25)].copy()
    for metric, ylabel, title, filename in [
        ("ap50_voc101", "AP50", "Resolution vs. AP50 at confidence 0.25", "fig01_ap50_vs_resolution"),
        ("recall", "Recall", "Resolution vs. recall at confidence 0.25", "fig02_recall_vs_resolution"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for model in MODEL_ORDER:
            g = primary[primary["model"].eq(model)].sort_values("resolution_width")
            ax.plot(
                g["resolution_width"],
                g[metric],
                marker=MARKERS[model],
                lw=2.2,
                color=COLORS[model],
                label=model,
            )
        ax.axhline(0.95, color="#111827", lw=1, ls="--", alpha=0.7)
        ax.axvspan(240, 288, color="#f59e0b", alpha=0.12)
        ax.set_xlabel("Frame width after downsampling (px)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.15, 1.03)
        ax.set_title(title)
        ax.grid(True, axis="both", alpha=0.25)
        ax.set_xticks([128, 160, 192, 240, 256, 288, 320, 384, 480, 640])
        ax.tick_params(axis="x", rotation=35)
        ax.legend(loc="lower right", fontsize=8)
        save_figure(fig, filename)


def make_figure_tradeoff(resolution: pd.DataFrame) -> None:
    primary = resolution[resolution["conf_threshold"].eq(0.25)].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for model in MODEL_ORDER:
        g = primary[primary["model"].eq(model)].sort_values("resolution_width")
        ax.plot(
            g["ap50_voc101"],
            g["fps"],
            color=COLORS[model],
            lw=2.8,
            alpha=0.86,
            label=model,
            zorder=2,
        )
    ax.axvline(0.95, color="#111827", lw=1, ls="--", alpha=0.7)
    ax.text(0.952, 29.5, "AP50 = 0.95", rotation=90, va="bottom", ha="left", fontsize=8, color="#374151")
    ax.set_xlabel("Accuracy (AP50 at IoU = 0.50)")
    ax.set_ylabel("Throughput (frames per second)")
    ax.set_title("Accuracy-throughput trade-off at confidence 0.25", pad=10)
    ax.set_xlim(0.18, 1.03)
    ax.set_ylim(28, 87.5)
    ax.grid(True, alpha=0.22, zorder=0)
    ax.legend(loc="lower left", fontsize=8, frameon=True, title="Detector")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "fig02_accuracy_fps_tradeoff")


def make_figure_threshold_heatmap(resolution: pd.DataFrame) -> None:
    widths = sorted(resolution["resolution_width"].unique())
    thresholds = sorted(resolution["conf_threshold"].unique())
    filenames = {
        "YOLOv5n-FT": "fig04_threshold_f1_yolov5n",
        "YOLOv8n-FT": "fig05_threshold_f1_yolov8n",
        "YOLOv12n-FT": "fig06_threshold_f1_yolov12n",
        "SCRFD": "fig07_threshold_f1_scrfd",
    }
    for model in MODEL_ORDER:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        g = resolution[resolution["model"].eq(model)]
        pivot = g.pivot_table(index="conf_threshold", columns="resolution_width", values="f1", aggfunc="mean").reindex(index=thresholds, columns=widths)
        im = ax.imshow(pivot.values, aspect="auto", vmin=0.2, vmax=1.0, cmap="viridis", origin="lower")
        ax.set_title(f"F1 threshold sensitivity: {model}")
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels([str(w) for w in widths], rotation=45)
        ax.set_yticks(range(len(thresholds)))
        ax.set_yticklabels([fmt(t, 2) for t in thresholds])
        ax.set_xlabel("Frame width (px)")
        ax.set_ylabel("Confidence threshold")
        cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
        cbar.set_label("F1 score")
        save_figure(fig, filenames[model])


def make_figure_error_modes(resolution: pd.DataFrame) -> None:
    primary = resolution[resolution["conf_threshold"].eq(0.25)].copy()
    primary["miss_rate"] = primary["fn"] / primary["sample_size"]
    primary["false_positives_per_100_frames"] = primary["fp"] / primary["sample_size"] * 100.0
    for metric, ylabel, title, filename in [
        ("miss_rate", "False negative rate", "Missed detections vs. resolution", "fig08_false_negative_rate"),
        ("false_positives_per_100_frames", "False positives per 100 frames", "False alarms vs. resolution", "fig09_false_positives_per_100_frames"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for model in MODEL_ORDER:
            g = primary[primary["model"].eq(model)].sort_values("resolution_width")
            ax.plot(
                g["resolution_width"],
                g[metric],
                marker=MARKERS[model],
                lw=2.0,
                color=COLORS[model],
                label=model,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Frame width after downsampling (px)")
        ax.set_xticks([128, 160, 192, 240, 256, 288, 320, 384, 480, 640])
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        save_figure(fig, filename)


def make_figure_face_scale(resolution: pd.DataFrame, manifest: pd.DataFrame) -> None:
    primary = resolution[resolution["conf_threshold"].eq(0.25)].copy()
    median_face_width = float(manifest[manifest["p1_indoor_split"].eq("eval")]["box_w"].median())
    primary["median_face_width_after_downscale"] = primary["resolution_width"] / 800.0 * median_face_width
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for model in MODEL_ORDER:
        g = primary[primary["model"].eq(model)].sort_values("median_face_width_after_downscale")
        ax.plot(g["median_face_width_after_downscale"], g["ap50_voc101"], marker=MARKERS[model], lw=2.2, color=COLORS[model], label=model)
    ax.axhline(0.95, color="#111827", lw=1, ls="--", alpha=0.7)
    ax.set_xlabel("Median derived face-box width after downsampling (px)")
    ax.set_ylabel("AP50")
    ax.set_title("Effective face scale explains the resolution cutoff")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, "fig05_effective_face_scale")


def make_figure_cutoff(resolution: pd.DataFrame) -> None:
    rows = build_cutoff_rows(resolution)
    labels = [r[0] for r in rows]
    widths = [0 if r[1] == "None" else int(r[1].split("x")[0]) for r in rows]
    fps = [0 if r[5] == "-" else float(r[5]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.bar(labels, widths, color=[COLORS[m] for m in labels], alpha=0.9)
    ax.set_ylabel("Minimum acceptable frame width (px)")
    ax.set_title("Smallest resolution satisfying AP50 >= 0.95 and recall >= 0.95")
    ax.set_ylim(0, 340)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, width, speed in zip(bars, widths, fps):
        ax.text(bar.get_x() + bar.get_width() / 2, width + 8, f"{width}px\n{speed:.1f} FPS", ha="center", va="bottom", fontsize=8)
    save_figure(fig, "fig06_minimum_acceptable_width")


def generate_figures(fixed: pd.DataFrame, resolution: pd.DataFrame, manifest: pd.DataFrame) -> None:
    del fixed
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    make_figure_accuracy_resolution(resolution)
    make_figure_tradeoff(resolution)
    make_figure_threshold_heatmap(resolution)
    make_figure_error_modes(resolution)
    make_figure_face_scale(resolution, manifest)
    make_figure_cutoff(resolution)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def latex_table_rows(rows: list[list[str]]) -> str:
    return "\n".join(" & ".join(latex_escape(cell) for cell in row) + r" \\" for row in rows)


def write_ieeetran_tex(fixed: pd.DataFrame, resolution: pd.DataFrame) -> None:
    fixed_rows = build_fixed_rows(fixed)
    cutoff_rows = build_cutoff_rows(resolution)
    fig_dir = "chokepoint_data/modeling/ieee_paper_figures_en"
    tex = rf"""\documentclass[conference]{{IEEEtran}}
\IEEEoverridecommandlockouts
\usepackage{{cite}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{url}}
\usepackage{{xcolor}}

\def\BibTeX{{{{\rm B\kern-.05em{{\sc i\kern-.025em b}}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{{E}}\kern-.125emX}}}}

\begin{{document}}

\title{{Resolution, Accuracy, and Throughput Trade-off for Lightweight Face Detection in ChokePoint Surveillance Video}}

\author{{
\IEEEauthorblockN{{Vu Hai Minh}}
\IEEEauthorblockA{{\textit{{Dept. of Artificial Intelligence}}\\
\textit{{FPT University}}\\
Ha Noi, Viet Nam\\
vhaiminh.2k5@gmail.com}}
\and
\IEEEauthorblockN{{Cao Van Mai}}
\IEEEauthorblockA{{\textit{{Dept. of Artificial Intelligence}}\\
\textit{{FPT University}}\\
Ha Noi, Viet Nam\\
maicv2@fe.edu.vn}}
}}

\maketitle

\begin{{abstract}}
Real-time face-recognition pipelines deployed on security cameras can fail when the number of visible faces increases suddenly, because every detected face creates additional localization, alignment, and recognition work. This paper studies a controllable mitigation strategy: reducing the input frame resolution until throughput improves while detection accuracy remains within an acceptable range. Using the ChokePoint surveillance dataset, we construct a detector-oriented benchmark from full-frame Portal 1 indoor sequences and derive face boxes from manually labelled eye landmarks. YOLOv5n, YOLOv8n, and YOLOv12n are fine-tuned on 6,035 training frames and compared with the SCRFD face detector on the full 5,676-frame evaluation split. The benchmark runs on an NVIDIA RTX 3050 Laptop GPU across ten downsampled widths from 640 to 128 pixels and six confidence thresholds from 0.25 to 0.70. At 640$\times$480, all fine-tuned YOLO models reach AP$_{{50}}=1.000$, while SCRFD reaches AP$_{{50}}=0.988$. The smallest width preserving both AP$_{{50}}\geq0.95$ and recall $\geq0.95$ is 240 pixels for YOLOv5n-FT, 288 pixels for YOLOv8n-FT and YOLOv12n-FT, and 256 pixels for SCRFD. The results identify 240--288 px as the practical accuracy-throughput operating band for this indoor choke-point protocol.
\end{{abstract}}

\begin{{IEEEkeywords}}
face detection, face recognition pipeline, surveillance video, ChokePoint, resolution scaling, throughput, YOLOv5, YOLOv8, YOLOv12, SCRFD
\end{{IEEEkeywords}}

\section{{Introduction}}
Face recognition systems used in access control and security monitoring are usually built as a pipeline: faces are detected in the camera frame, cropped and aligned, then passed to a recognition or verification model. In normal office-door conditions this pipeline may be stable, but a sudden increase in the number of faces in the frame can multiply the downstream workload and cause the system to miss real-time deadlines or fail operationally.

The motivating failure mode for this study came from an SCRFD-based security camera deployment in which the recognition service became unreliable when many faces appeared in the scene at once. This observation suggests that model selection alone is not enough. A deployment needs an operating point that balances three coupled variables: frame resolution, detector accuracy, and throughput.

Input resolution is a direct and controllable variable. Lower resolution reduces image memory, preprocessing cost, and detector workload, but it also compresses small faces until localization fails. Higher resolution preserves facial structure but reduces frame rate and can overload the recognition pipeline during crowded intervals. The practical question is therefore: how low can the frame resolution be reduced before the detector no longer provides reliable face crops?

\section{{Related Work}}
The accuracy-throughput problem has been central to face detection since the early real-time detector literature. Viola and Jones framed face detection as a practical real-time task and used a cascade to reject most image windows quickly while preserving detection quality \cite{{viola2004robust}}. Low-resolution face analysis is especially important in surveillance. Li et al. show that low-resolution face recognition remains challenging under non-ideal surveillance-like conditions such as blur, lighting variation, non-frontal pose, and small face size \cite{{li2018lowreswild}}.

YOLO introduced real-time single-stage object detection by predicting boxes and class probabilities in one forward pass \cite{{redmon2016yolo}}. YOLOv5 and YOLOv8 are widely used through the Ultralytics ecosystem, while YOLOv12 introduces an attention-centric design for real-time object detection \cite{{jocher2020yolov5,reis2023yolov8,tian2025yolov12}}. SCRFD is a specialized face detector designed around sample redistribution and computation redistribution to improve the efficiency-accuracy trade-off for face detection \cite{{guo2021scrfd}}. ChokePoint is selected because it was introduced for video-based face recognition in surveillance-like portal conditions \cite{{wong2011chokepoint}}.

\section{{Dataset and Label Protocol}}
ChokePoint contains full-frame camera images, cropped face images, and XML ground truth with manually labelled eye coordinates. This benchmark uses the full-frame camera images rather than the 96$\times$96 cropped face images, because the goal is to evaluate whether a detector can find faces in the original surveillance frame.

The local protocol follows the Portal 1 indoor frontal split. The training set contains 6,035 frames and the evaluation set contains 5,676 frames. Ground-truth face boxes are derived from eye landmarks. If $d$ is the inter-eye distance, the box width is set to $2.8d$ and the box height to $3.6d$, centered around the eye midpoint with a vertical offset that covers forehead, eyes, nose, mouth, and chin.

\begin{{table}}[htbp]
\caption{{Experimental protocol.}}
\label{{tab:protocol}}
\centering
\begin{{tabular}}{{p{{0.34\linewidth}}p{{0.56\linewidth}}}}
\toprule
Item & Value \\
\midrule
Dataset & ChokePoint Portal 1 indoor frontal split \\
Train/evaluation & 6,035 / 5,676 full-frame images \\
Models & YOLOv5n-FT, YOLOv8n-FT, YOLOv12n-FT, SCRFD \\
Frame widths & 640, 480, 384, 320, 288, 256, 240, 192, 160, 128 px \\
Confidence thresholds & 0.25, 0.35, 0.45, 0.55, 0.65, 0.70 \\
Hardware & NVIDIA GeForce RTX 3050 Laptop GPU \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Experimental Setup}}
YOLOv5n, YOLOv8n, and YOLOv12n are initialized from nano-scale weights and fine-tuned for 20 epochs at image size 640 on the ChokePoint training split. SCRFD is evaluated through InsightFace using the buffalo\_l detector module. A detection is counted as a true positive if it matches an unmatched derived ground-truth box with intersection-over-union at least 0.50. Because missed detections are usually more damaging than extra face proposals in a recognition pipeline, the deployment cutoff requires both AP$_{{50}}\geq0.95$ and recall $\geq0.95$.

\section{{Results}}
\begin{{table*}}[htbp]
\caption{{Fixed-resolution benchmark at 640$\times$480.}}
\label{{tab:fixed}}
\centering
\begin{{tabular}}{{llrrrrr}}
\toprule
Model & Device & Prec. & Recall & F1 & AP50 & FPS \\
\midrule
{latex_table_rows(fixed_rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}

Table~\ref{{tab:fixed}} confirms the need for face-specific adaptation. The generic COCO-pretrained YOLO nano models produce effectively zero AP$_{{50}}$ under the face-box protocol. After fine-tuning, all three YOLO variants reach perfect AP$_{{50}}$ at 640$\times$480. SCRFD transfers without dataset-specific training and remains a strong specialized baseline.

\begin{{figure*}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{{fig_dir}/fig01_accuracy_recall_vs_resolution.pdf}}
\caption{{AP$_{{50}}$ and recall as frame width is reduced at confidence 0.25.}}
\label{{fig:resolution}}
\end{{figure*}}

\begin{{figure*}}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{{{fig_dir}/fig02_accuracy_fps_tradeoff.pdf}}
\caption{{Accuracy-throughput trade-off. The horizontal axis is AP$_{{50}}$ and the vertical axis is frames per second.}}
\label{{fig:tradeoff}}
\end{{figure*}}

\begin{{figure*}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{{fig_dir}/fig03_threshold_f1_heatmap.pdf}}
\caption{{F1 sensitivity to confidence threshold and resolution.}}
\label{{fig:threshold}}
\end{{figure*}}

\begin{{figure*}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{{fig_dir}/fig04_error_modes_vs_resolution.pdf}}
\caption{{False negative and false positive behavior as resolution changes.}}
\label{{fig:error}}
\end{{figure*}}

\begin{{table}}[htbp]
\caption{{Smallest resolution satisfying AP$_{{50}}\geq0.95$ and recall $\geq0.95$.}}
\label{{tab:cutoff}}
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
Detector & Width & AP50 & Recall & FPS \\
\midrule
"""
    for model, cutoff, _label, ap50, recall, fps in cutoff_rows:
        width = cutoff.split("x")[0] if cutoff != "None" else "-"
        tex += f"{latex_escape(model)} & {latex_escape(width)} & {ap50} & {recall} & {fps} " + r"\\" + "\n"

    tex += rf"""\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Deployment Recommendations}}
For favorable conditions where accuracy is the priority and GPU throughput is available, YOLOv8n-FT at 480$\times$360 or 640$\times$480 is recommended. It preserves AP$_{{50}}=1.000$ while running faster than YOLOv5n-FT and YOLOv12n-FT in the full-resolution evaluation. For constrained but reliable operation, YOLOv8n-FT at 288$\times$216 or 320$\times$240 is the most balanced choice. If the system must use the smallest acceptable frame, YOLOv5n-FT at 240$\times$180 is viable but leaves less safety margin. For poor conditions or extreme downsampling, SCRFD is more stable, but it should be treated as a fallback rather than a recommended recognition setting below the AP$_{{50}}$ target.

\section{{Discussion}}
The most important result is that the optimal resolution is not simply the lowest or highest tested value. The useful region is a band where faces are still large enough for reliable localization and the detector is still fast enough to reduce recognition-pipeline pressure. In this protocol, that band is 240--288 px width.

YOLOv12n-FT performs worse than expected for this specific deployment problem. It reaches perfect AP$_{{50}}$ at high resolution, but it is slower than YOLOv5n-FT and YOLOv8n-FT and loses accuracy sharply under strong downsampling. This suggests that attention-centric design does not automatically translate into better small-face robustness after resolution collapse.

\section{{Limitations and Future Work}}
The first limitation is the label protocol. ChokePoint provides eye landmarks, not native detector boxes, so this paper evaluates against deterministically derived boxes. The second limitation is scope. The current benchmark intentionally focuses on Portal 1 indoor frontal sequences because they match the office-door security-camera motivation. Future work should extend the protocol to Portal 2, crowded sequences, cross-camera transfer, and end-to-end recognition accuracy after detection.

\section{{Conclusion}}
This paper presented a ChokePoint-only benchmark for selecting the minimum useful resolution for lightweight face detection in surveillance video. The results show that 240--288 px frame width is the practical operating band for preserving detection quality while improving throughput. YOLOv8n-FT is the strongest balanced recommendation, YOLOv5n-FT provides the smallest acceptable resolution at 240$\times$180, and SCRFD remains the most robust fallback under extremely low resolution.

\begin{{thebibliography}}{{00}}
\bibitem{{viola2004robust}} P. Viola and M. J. Jones, ``Robust real-time face detection,'' \textit{{Int. J. Comput. Vis.}}, vol. 57, no. 2, pp. 137--154, 2004.
\bibitem{{li2018lowreswild}} P. Li, L. Prieto, D. Mery, and P. J. Flynn, ``On low-resolution face recognition in the wild: Comparisons and new techniques,'' arXiv:1805.11529, 2018.
\bibitem{{redmon2016yolo}} J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, ``You Only Look Once: Unified, real-time object detection,'' in \textit{{Proc. IEEE Conf. Comput. Vis. Pattern Recognit.}}, 2016, pp. 779--788.
\bibitem{{jocher2020yolov5}} G. Jocher et al., ``ultralytics/yolov5,'' GitHub repository and software release, 2020--2022.
\bibitem{{reis2023yolov8}} D. Reis, J. Kupec, J. Hong, and A. Daoudi, ``Real-time flying object detection with YOLOv8,'' arXiv:2305.09972, 2023.
\bibitem{{tian2025yolov12}} Y. Tian, Q. Ye, and D. Doermann, ``YOLOv12: Attention-centric real-time object detectors,'' arXiv:2502.12524, 2025.
\bibitem{{guo2021scrfd}} J. Guo, J. Deng, A. Lattas, and S. Zafeiriou, ``Sample and computation redistribution for efficient face detection,'' arXiv:2105.04714, 2021.
\bibitem{{wong2011chokepoint}} Y. Wong, S. Chen, S. Mau, C. Sanderson, and B. C. Lovell, ``Patch-based probabilistic image quality assessment for face selection and improved video-based face recognition,'' in \textit{{Proc. IEEE Biometrics Workshop, CVPR Workshops}}, 2011, pp. 81--88.
\end{{thebibliography}}

\end{{document}}
"""
    TEX_OUTPUT.write_text(tex, encoding="utf-8")


def build_docx() -> None:
    fixed, resolution, manifest = load_data()
    generate_figures(fixed, resolution, manifest)
    doc = Document()
    set_document_defaults(doc)

    add_centered(doc, TITLE, 18, True)
    add_author_block(doc)
    add_abstract(
        doc,
        " Real-time face-recognition pipelines deployed on security cameras can fail "
        "when the number of visible faces increases suddenly, because every detected "
        "face creates additional localization, alignment, and recognition work. This "
        "paper studies a controllable mitigation strategy: reducing the input frame "
        "resolution until throughput improves while detection accuracy remains within "
        "an acceptable range. Using the ChokePoint surveillance dataset, we construct "
        "a detector-oriented benchmark from full-frame Portal 1 indoor sequences and "
        "derive face boxes from manually labelled eye landmarks. YOLOv5n, YOLOv8n, "
        "and YOLOv12n are fine-tuned on 6,035 training frames and compared with the "
        "SCRFD face detector on the full 5,676-frame evaluation split. The benchmark "
        "runs on an NVIDIA RTX 3050 Laptop GPU across ten downsampled widths from "
        "640 to 128 pixels and six confidence thresholds from 0.25 to 0.70. At "
        "640x480, all fine-tuned YOLO models reach AP50 = 1.000, while SCRFD reaches "
        "AP50 = 0.988. The smallest width preserving both AP50 >= 0.95 and recall "
        ">= 0.95 is 240 pixels for YOLOv5n-FT, 288 pixels for YOLOv8n-FT and "
        "YOLOv12n-FT, and 256 pixels for SCRFD. The results identify 240-288 px as "
        "the practical accuracy-throughput operating band for this indoor choke-point "
        "protocol, with YOLOv8n-FT preferred for balanced deployment, YOLOv5n-FT for "
        "the smallest acceptable resolution, and SCRFD as a robust fallback when "
        "input resolution becomes extremely low.",
        " face detection, face recognition pipeline, surveillance video, ChokePoint, "
        "resolution scaling, throughput, YOLOv5, YOLOv8, YOLOv12, SCRFD.",
    )

    configure_section(doc.add_section(WD_SECTION.CONTINUOUS), columns=2)

    add_ieee_heading(doc, "I", "Introduction")
    add_body(
        doc,
        "Face recognition systems used in access control and security monitoring are "
        "usually built as a pipeline: faces are detected in the camera frame, cropped "
        "and aligned, then passed to a recognition or verification model. In normal "
        "office-door conditions this pipeline may be stable, but a sudden increase in "
        "the number of faces in the frame can multiply the downstream workload and "
        "cause the system to miss real-time deadlines or fail operationally."
    )
    add_body(
        doc,
        "The motivating failure mode for this study came from an SCRFD-based security "
        "camera deployment in which the recognition service became unreliable when "
        "many faces appeared in the scene at once. This observation suggests that "
        "model selection alone is not enough. A deployment needs an operating point "
        "that balances three coupled variables: frame resolution, detector accuracy, "
        "and throughput."
    )
    add_body(
        doc,
        "Input resolution is a direct and controllable variable. Lower resolution "
        "reduces image memory, preprocessing cost, and detector workload, but it also "
        "compresses small faces until localization fails. Higher resolution preserves "
        "facial structure but reduces frame rate and can overload the recognition "
        "pipeline during crowded intervals. The practical question is therefore: how "
        "low can the frame resolution be reduced before the detector no longer "
        "provides reliable face crops?"
    )
    add_body(
        doc,
        "This paper answers that question on ChokePoint, a surveillance dataset "
        "recorded at pedestrian portals. The contribution is a ChokePoint-only "
        "benchmark that converts eye landmarks into deterministic detector boxes, "
        "fine-tunes lightweight YOLO detectors, compares them against SCRFD, and "
        "maps the accuracy-throughput curve across multiple standard-like frame "
        "resolutions."
    )

    add_ieee_heading(doc, "II", "Related Work")
    add_body(
        doc,
        "The accuracy-throughput problem has been central to face detection since the "
        "early real-time detector literature. Viola and Jones framed face detection "
        "as a practical real-time task and used a cascade to reject most image windows "
        "quickly while preserving detection quality [1]. Although modern detectors "
        "are deep neural networks rather than Haar cascades, the deployment question "
        "is similar: a security-camera system must localize faces fast enough to keep "
        "the downstream recognition pipeline stable."
    )
    add_body(
        doc,
        "Low-resolution face analysis is especially important in surveillance. Li "
        "et al. show that low-resolution face recognition remains challenging when "
        "faces are captured under non-ideal surveillance-like conditions such as "
        "blur, lighting variation, non-frontal pose, and small face size [2]. Their "
        "analysis motivates treating resolution as an explicit deployment variable "
        "rather than as a fixed preprocessing detail."
    )
    add_body(
        doc,
        "YOLO introduced real-time single-stage object detection by predicting boxes "
        "and class probabilities in one forward pass [3]. The YOLOv5n, YOLOv8n, and "
        "YOLOv12n nano models in this study are treated as lightweight descendants "
        "of this real-time detection family. The generic COCO-pretrained YOLO nano "
        "models are used as a sanity check only; fair comparison requires fine-tuning "
        "on ChokePoint-derived face boxes."
    )
    add_body(
        doc,
        "SCRFD is a specialized face detector designed around sample redistribution "
        "and computation redistribution to improve the efficiency-accuracy trade-off "
        "for face detection. It is therefore a strong baseline for surveillance "
        "deployment. ChokePoint is selected as the dataset because it was introduced "
        "for video-based face recognition in surveillance-like portal conditions [4] "
        "and is publicly distributed as a documented surveillance dataset record [5]."
    )

    add_ieee_heading(doc, "III", "Dataset and Label Protocol")
    add_body(
        doc,
        "ChokePoint contains full-frame camera images, cropped face images, and XML "
        "ground truth with manually labelled eye coordinates. This benchmark uses the "
        "full-frame camera images rather than the 96x96 cropped face images, because "
        "the goal is to evaluate whether a detector can find faces in the original "
        "surveillance frame."
    )
    add_body(
        doc,
        "The local protocol follows the Portal 1 indoor frontal split. The training "
        "set contains 6,035 frames and the evaluation set contains 5,676 frames. "
        "Ground-truth face boxes are derived from eye landmarks. If d is the inter-eye "
        "distance, the box width is set to 2.8d and the box height to 3.6d, centered "
        "around the eye midpoint with a vertical offset that covers forehead, eyes, "
        "nose, mouth, and chin. Coordinates are clipped to the 800x600 frame."
    )
    add_body(
        doc,
        "This conversion is deterministic and reproducible, but it is also a "
        "limitation. The reported IoU and AP values measure agreement with derived "
        "face boxes rather than manually drawn face rectangles. For that reason, "
        "AP50, recall, F1, and throughput are emphasized over stricter localization "
        "metrics."
    )

    add_table(
        doc,
        "TABLE I. Experimental protocol.",
        ["Item", "Value"],
        [
            ["Dataset", "ChokePoint Portal 1 indoor frontal split"],
            ["Train / evaluation", "6,035 / 5,676 full-frame images"],
            ["Models", "YOLOv5n-FT, YOLOv8n-FT, YOLOv12n-FT, SCRFD"],
            ["Frame widths", "640, 480, 384, 320, 288, 256, 240, 192, 160, 128 px"],
            ["Confidence thresholds", "0.25, 0.35, 0.45, 0.55, 0.65, 0.70"],
            ["Hardware", "NVIDIA GeForce RTX 3050 Laptop GPU"],
        ],
    )

    add_ieee_heading(doc, "IV", "Experimental Setup")
    add_body(
        doc,
        "YOLOv5n, YOLOv8n, and YOLOv12n are initialized from nano-scale weights and "
        "fine-tuned for 20 epochs at image size 640 on the ChokePoint training split. "
        "SCRFD is evaluated through InsightFace using the buffalo_l detector module. "
        "The detector input is downsampled to each target width while preserving the "
        "original 4:3 aspect ratio; predicted boxes are then scaled back to the "
        "original 800x600 coordinate system for evaluation."
    )
    add_body(
        doc,
        "A detection is counted as a true positive if it matches an unmatched derived "
        "ground-truth box with IoU at least 0.50. The main metrics are precision, "
        "recall, F1, AP50, mean matched IoU, milliseconds per image, and frames per "
        "second. Because missed detections are usually more damaging than extra face "
        "proposals in a recognition pipeline, the deployment cutoff requires both "
        "AP50 >= 0.95 and recall >= 0.95."
    )

    add_ieee_heading(doc, "V", "Results")
    add_table(
        doc,
        "TABLE II. Fixed-resolution benchmark at 640x480.",
        ["Model", "Recall", "AP50", "FPS"],
        build_fixed_rows_compact(fixed),
        wide=True,
    )
    add_body(
        doc,
        "Table II first confirms the need for face-specific adaptation. The generic "
        "COCO-pretrained YOLO nano models produce effectively zero AP50 under the "
        "face-box protocol, because their class set and localization convention do "
        "not match ChokePoint faces. After fine-tuning, all three YOLO variants reach "
        "perfect AP50 at 640x480. SCRFD transfers without dataset-specific training "
        "and remains a strong specialized baseline, but it is slower on this GPU."
    )

    add_figure(
        doc,
        "fig01_ap50_vs_resolution.png",
        "Fig. 1. AP50 as frame width is reduced at confidence 0.25.",
        "Insight: AP50 stays near the deployment target until the 240-288 px band, "
        "then drops sharply for the fine-tuned YOLO models. YOLOv5n-FT reaches the "
        "most aggressive cutoff at 240x180, while YOLOv8n-FT and YOLOv12n-FT require "
        "288x216 for the same target.",
    )
    add_figure(
        doc,
        "fig02_recall_vs_resolution.png",
        "Fig. 2. Recall as frame width is reduced at confidence 0.25.",
        "Insight: recall is the metric that exposes low-resolution failure most "
        "clearly. Below 192 px, YOLO recall collapses, while SCRFD degrades more "
        "gradually but still falls below the recommended operating target.",
    )
    add_figure(
        doc,
        "fig02_accuracy_fps_tradeoff.png",
        "Fig. 3. Accuracy-throughput trade-off; the horizontal axis is AP50 and the vertical axis is FPS.",
        "Insight: points in the upper-right corner are preferred. YOLOv8n-FT is the "
        "best general recommendation when the system can use 288-640 px frames, "
        "because it combines near-perfect AP50 with the highest throughput among the "
        "high-accuracy detectors. SCRFD is attractive only when very low resolution "
        "or conservative face proposals matter more than peak AP50.",
    )
    add_figure(
        doc,
        "fig04_threshold_f1_yolov5n.png",
        "Fig. 4. F1 sensitivity of YOLOv5n-FT to confidence threshold and resolution.",
        "Insight: YOLOv5n-FT keeps high F1 down to 240 px when the threshold is not "
        "too strict. At 192 px and below, lowering the threshold cannot recover the "
        "lost face detail.",
    )
    add_figure(
        doc,
        "fig05_threshold_f1_yolov8n.png",
        "Fig. 5. F1 sensitivity of YOLOv8n-FT to confidence threshold and resolution.",
        "Insight: YOLOv8n-FT is the most balanced detector in the recommended region. "
        "It remains stable from 288 px upward and is therefore the safest general "
        "choice when both accuracy and throughput matter.",
    )
    add_figure(
        doc,
        "fig06_threshold_f1_yolov12n.png",
        "Fig. 6. F1 sensitivity of YOLOv12n-FT to confidence threshold and resolution.",
        "Insight: YOLOv12n-FT is more fragile under aggressive downsampling. Its F1 "
        "drops faster than YOLOv5n-FT and YOLOv8n-FT, which supports treating YOLOv12n "
        "as a topic for further tuning rather than the default deployment model.",
    )
    add_figure(
        doc,
        "fig07_threshold_f1_scrfd.png",
        "Fig. 7. F1 sensitivity of SCRFD to confidence threshold and resolution.",
        "Insight: SCRFD is less sensitive to threshold changes and remains more stable "
        "at very low resolution. Its role is strongest as a robust fallback when "
        "input frames are heavily downsampled.",
    )
    add_figure(
        doc,
        "fig08_false_negative_rate.png",
        "Fig. 8. False negative rate as frame width is reduced.",
        "Insight: the main low-resolution failure mode is missed faces. This matters "
        "for recognition because a missed face never reaches the identity model, so "
        "deployment should use a recall-constrained cutoff.",
    )
    add_figure(
        doc,
        "fig09_false_positives_per_100_frames.png",
        "Fig. 9. False positives per 100 frames as frame width is reduced.",
        "Insight: false alarms are not the dominant problem at the recommended "
        "operating widths. The resolution decision should therefore prioritize "
        "preserving recall over only minimizing false positives.",
    )
    add_figure(
        doc,
        "fig05_effective_face_scale.png",
        "Fig. 10. AP50 as a function of median derived face-box width after downsampling.",
        "Insight: the resolution cutoff is really a face-scale cutoff. The drop begins "
        "when downsampling pushes the median face box into a range where the detector "
        "has too few pixels to preserve stable facial structure. This supports using "
        "adaptive resolution or face-size monitoring in future deployments.",
    )
    add_figure(
        doc,
        "fig06_minimum_acceptable_width.png",
        "Fig. 11. Smallest frame width satisfying AP50 >= 0.95 and recall >= 0.95.",
        "Insight: the safest all-around setting is 288x216 with YOLOv8n-FT, while "
        "240x180 with YOLOv5n-FT is the most compressed acceptable configuration. "
        "SCRFD is not the fastest at high resolution, but it is useful as a robust "
        "fallback when frames are heavily downsampled.",
    )

    add_ieee_heading(doc, "VI", "Deployment Recommendations")
    add_body(
        doc,
        "For favorable conditions where accuracy is the priority and GPU throughput "
        "is available, YOLOv8n-FT at 480x360 or 640x480 is recommended. It preserves "
        "AP50 = 1.000 while running faster than YOLOv5n-FT and YOLOv12n-FT in the "
        "full-resolution evaluation."
    )
    add_body(
        doc,
        "For constrained but still reliable operation, YOLOv8n-FT at 288x216 or "
        "320x240 is the most balanced choice. This range remains above the AP50 and "
        "recall targets while maintaining roughly 60-69 FPS in the resolution sweep. "
        "If the system must use the smallest acceptable frame, YOLOv5n-FT at 240x180 "
        "is viable but leaves less safety margin."
    )
    add_body(
        doc,
        "For poor conditions or extreme downsampling, SCRFD is the more stable "
        "detector. At 128-192 px, SCRFD keeps substantially higher recall and AP50 "
        "than the fine-tuned YOLO variants. However, it falls below the AP50 >= 0.95 "
        "target at those widths, so it should be treated as a fallback rather than a "
        "recommended production setting for recognition."
    )

    add_ieee_heading(doc, "VII", "Discussion")
    add_body(
        doc,
        "The most important result is that the optimal resolution is not simply the "
        "lowest or highest tested value. The useful region is a band where faces are "
        "still large enough for reliable localization and the detector is still fast "
        "enough to reduce recognition-pipeline pressure. In this protocol, that band "
        "is 240-288 px width."
    )
    add_body(
        doc,
        "YOLOv12n-FT performs worse than expected for this specific deployment "
        "problem. It reaches perfect AP50 at high resolution, but it is slower than "
        "YOLOv5n-FT and YOLOv8n-FT and loses accuracy sharply under strong "
        "downsampling. This suggests that attention-centric design does not "
        "automatically translate into better small-face robustness after resolution "
        "collapse. Future work should test whether longer training, augmentation for "
        "low-resolution faces, or architecture-specific hyperparameter tuning can "
        "close this gap."
    )
    add_body(
        doc,
        "The study also clarifies the role of fine-tuning. COCO-pretrained YOLO "
        "models are useful initial weights, but not valid face detectors for this "
        "benchmark. Once fine-tuned on ChokePoint-derived boxes, YOLOv5n and YOLOv8n "
        "become strong deployment candidates and can outperform SCRFD in the "
        "high-accuracy operating region."
    )

    add_ieee_heading(doc, "VIII", "Limitations and Future Work")
    add_body(
        doc,
        "The first limitation is the label protocol. ChokePoint provides eye "
        "landmarks, not native detector boxes, so this paper evaluates against "
        "deterministically derived boxes. A future version should manually annotate a "
        "small validation subset to quantify the sensitivity of the box conversion."
    )
    add_body(
        doc,
        "The second limitation is scope. The current benchmark intentionally focuses "
        "on Portal 1 indoor frontal sequences because they match the office-door "
        "security-camera motivation. Future work should extend the protocol to "
        "Portal 2, crowded sequences, cross-camera transfer, and end-to-end "
        "recognition accuracy after detection. The most promising extension is an "
        "adaptive policy that chooses frame resolution based on current face count, "
        "face scale, and GPU load."
    )

    add_ieee_heading(doc, "IX", "Conclusion")
    add_body(
        doc,
        "This paper presented a ChokePoint-only benchmark for selecting the minimum "
        "useful resolution for lightweight face detection in surveillance video. "
        "The results show that 240-288 px frame width is the practical operating "
        "band for preserving detection quality while improving throughput. YOLOv8n-FT "
        "is the strongest balanced recommendation, YOLOv5n-FT provides the smallest "
        "acceptable resolution at 240x180, and SCRFD remains the most robust fallback "
        "under extremely low resolution. These findings provide a concrete basis for "
        "reducing security-camera preprocessing cost before downstream face "
        "recognition."
    )

    add_references_heading(doc)
    references = [
        "[1] P. Viola and M. J. Jones, \"Robust real-time face detection,\" Int. J. Comput. Vis., vol. 57, no. 2, pp. 137-154, 2004, doi: 10.1023/B:VISI.0000013087.49260.fb.",
        "[2] P. Li, L. Prieto, D. Mery, and P. J. Flynn, \"On low-resolution face recognition in the wild: Comparisons and new techniques,\" arXiv:1805.11529, 2018.",
        "[3] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, \"You Only Look Once: Unified, real-time object detection,\" in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 779-788.",
        "[4] Y. Wong, S. Chen, S. Mau, C. Sanderson, and B. C. Lovell, \"Patch-based probabilistic image quality assessment for face selection and improved video-based face recognition,\" in Proc. IEEE Biometrics Workshop, Computer Vision and Pattern Recognition Workshops, 2011, pp. 81-88, doi: 10.1109/CVPRW.2011.5981881.",
        "[5] Y. Wong, S. Chen, S. Mau, C. Sanderson, and B. C. Lovell, \"ChokePoint Dataset,\" Zenodo, version v1, Jun. 22, 2017, doi: 10.5281/zenodo.815657.",
    ]
    for ref in references:
        add_reference(doc, ref)

    try:
        doc.save(OUTPUT)
        written = OUTPUT
    except PermissionError:
        doc.save(OUTPUT_FALLBACK)
        written = OUTPUT_FALLBACK
    print(f"Wrote {written}")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    build_docx()
