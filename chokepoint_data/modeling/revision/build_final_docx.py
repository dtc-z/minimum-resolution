from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


TITLE = "Minimum-Resolution Face Detection for Real-Time Surveillance: A Benchmark of YOLO and SCRFD"


def set_columns(section, count: int, spacing_twips: int) -> None:
    columns = section._sectPr.xpath("./w:cols")
    element = columns[0] if columns else OxmlElement("w:cols")
    element.set(qn("w:num"), str(count))
    element.set(qn("w:space"), str(spacing_twips))
    if not columns:
        section._sectPr.append(element)


def clear_body_preserve_final_section(document) -> None:
    body = document._element.body
    final_section = body.sectPr
    for child in list(body):
        if child is not final_section:
            body.remove(child)
    for section in document.sections:
        for paragraph in section.header.paragraphs:
            paragraph.text = ""
        for paragraph in section.footer.paragraphs:
            paragraph.text = ""


def set_table_borders(table, style: str = "grid") -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single" if style == "grid" else "nil")
        if style == "grid":
            element.set(qn("w:sz"), "4")
            element.set(qn("w:space"), "0")
            element.set(qn("w:color"), "808080")


def set_cell_text(cell, text: str, bold: bool = False, size: float = 7.5, align=None) -> None:
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    paragraph = cell.paragraphs[0]
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)


def add_table(document, headers: list[str], rows: list[list[str]], widths: list[float]):
    table = document.add_table(rows=1, cols=len(headers))
    table.autofit = False
    set_table_borders(table)
    for index, (header, width) in enumerate(zip(headers, widths)):
        table.columns[index].width = Inches(width)
        table.rows[0].cells[index].width = Inches(width)
        set_cell_text(table.rows[0].cells[index], header, bold=True)
    for values in rows:
        cells = table.add_row().cells
        for index, (value, width) in enumerate(zip(values, widths)):
            cells[index].width = Inches(width)
            set_cell_text(cells[index], value)
    document.add_paragraph()
    return table


def add_body(document, text: str, style="Body Text"):
    return document.add_paragraph(text, style=style)


def add_heading(document, text: str, level=1):
    style = "Heading 1" if level == 1 else "Heading 2"
    return document.add_paragraph(text, style=style)


def add_caption(document, text: str, table=False):
    style = "table head" if table else "figure caption"
    return document.add_paragraph(text, style=style)


def get_row(summary: pd.DataFrame, model: str, width: int, confidence: float):
    selected = summary[
        (summary["model"] == model)
        & (summary["resolution_width"] == width)
        & (summary["conf_threshold"] == confidence)
    ]
    if selected.empty:
        raise ValueError(f"No final test summary row for {model}, {width}px, confidence={confidence}")
    return selected.iloc[0]


def get_manual_row(summary: pd.DataFrame, model: str, width: int, confidence: float):
    selected = summary[
        (summary["model"] == model)
        & (summary["resolution_width"] == width)
        & (summary["conf_threshold"] == confidence)
    ]
    if selected.empty:
        raise ValueError(f"No manual-box result for {model}, {width}px, confidence={confidence}")
    return selected.iloc[0]


def make_abstract(summary: pd.DataFrame, selection: dict, manual_summary: pd.DataFrame,
                  manual_stats: dict) -> str:
    if selection.get("selected_model"):
        model = selection["selected_model"]
        width = int(selection["selected_width"])
        row = get_row(summary, model, width, float(selection["confidence"]))
        audit_row = get_manual_row(manual_summary, model, width, float(selection["confidence"]))
        selected_result = (
            f"The development-only rule selected {model} at {width} pixels. On the locked test set, "
            f"this setting obtained average precision at IoU 0.50 (AP50) {row['ap50_vs_derived_boxes']:.3f}, recall {row['recall']:.3f}, "
            f"and median throughput {row['fps_median']:.1f} frames per second. Against the latest user-designated "
            f"hand-drawn boxes on 240 test frames, the same setting obtained AP50 {audit_row['ap50']:.3f} and "
            f"recall {audit_row['recall']:.3f}."
        )
    else:
        selected_result = (
            "No model and resolution met the prespecified development-set AP50 and recall targets "
            "at confidence 0.25."
        )
    return (
        "This study measures the detector-level trade-off between input resolution, detection accuracy, "
        "and throughput for ChokePoint Portal 1 indoor frontal video. We split complete sequences into "
        "4,900 training, 1,135 development, and 5,676 held-out test frames, and compare fine-tuned "
        "YOLOv5nu, YOLOv8n, YOLOv12n, and SCRFD-500M. Evaluation sweeps ten input widths from 640 to "
        "128 pixels and six confidence thresholds; operating-point selection uses development data only. "
        + selected_result
        + f" The median IoU between the hand-drawn and landmark-derived boxes was "
        f"{manual_stats['manual_vs_proxy_iou']['50%']:.3f}. The benchmark and audit describe detector "
        "performance on one portal and GPU; they do not measure identity recognition "
        "or end-to-end pipeline performance."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the revised paper using the IEEE A4 conference template.")
    parser.add_argument("--template", default="docs/IEEE_Conference_Template_A4_2024_python_compatible.docx")
    parser.add_argument("--summary", default="chokepoint_data/modeling/final_revision/benchmark/summary.csv")
    parser.add_argument("--selection", default="chokepoint_data/modeling/final_revision/benchmark/operating_point_dev_locked.json")
    parser.add_argument("--figures", default="chokepoint_data/modeling/final_revision/figures")
    parser.add_argument("--manual-summary", default="final/benchmark/manual_box_audit_latest/manual_box_summary.csv")
    parser.add_argument("--manual-stats", default="final/benchmark/manual_box_audit_latest/manual_box_summary.json")
    parser.add_argument("--pseudo-summary", default="final/benchmark/pseudo_box_audit/pseudo_box_summary.csv")
    parser.add_argument("--out", default="docs/Paper_Improved_Final.docx")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = pd.read_csv(summary_path)
    if not summary["split"].eq("test").all() or summary.empty:
        raise ValueError("Canonical summary must contain only final test rows")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    manual_summary = pd.read_csv(args.manual_summary)
    manual_stats = json.loads(Path(args.manual_stats).read_text(encoding="utf-8"))
    pseudo_summary = pd.read_csv(args.pseudo_summary)
    figures = Path(args.figures)
    required_figures = [
        figures / "fig01_accuracy_by_resolution.png",
        figures / "fig02_throughput_by_resolution.png",
        figures / "fig03_threshold_sensitivity.png",
    ]
    for figure in required_figures:
        if not figure.is_file():
            raise FileNotFoundError(figure)

    template = Document(args.template)
    clear_body_preserve_final_section(template)
    section = template.sections[0]
    section.page_width = Inches(8.268)
    section.page_height = Inches(11.693)
    set_columns(section, 1, 36)

    title = template.add_paragraph(TITLE, style="paper title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_table = template.add_table(rows=1, cols=2)
    author_table.autofit = False
    set_table_borders(author_table, style="none")
    for cell in author_table.rows[0].cells:
        cell.width = Inches(3.2)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
    author_info = [
        ["Vu Hai Minh", "Department of Artificial Intelligence", "FPT University", "Ha Noi, Viet Nam", "vhaiminh.2k5@gmail.com"],
        ["Cao Van Mai", "Department of Artificial Intelligence", "FPT University", "Ha Noi, Viet Nam", "maicv2@fe.edu.vn"],
    ]
    for cell, lines in zip(author_table.rows[0].cells, author_info):
        cell.text = ""
        for index, line in enumerate(lines):
            paragraph = cell.paragraphs[0] if index == 0 else cell.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.style = template.styles["Author"]
            paragraph.add_run(line)

    body_section = template.add_section(WD_SECTION.CONTINUOUS)
    set_columns(body_section, 2, 18)

    abstract = template.add_paragraph(style="Abstract")
    abstract.add_run("Abstract— ").bold = True
    abstract.add_run(make_abstract(summary, selection, manual_summary, manual_stats))
    keywords = template.add_paragraph(style="Keywords")
    keywords.add_run("Index Terms— ").bold = True
    keywords.add_run("face detection, surveillance video, ChokePoint, resolution scaling, throughput, YOLO, SCRFD.")

    add_heading(template, "I. INTRODUCTION")
    add_body(template,
        "Video surveillance systems commonly place a detector before face alignment and identity recognition. "
        "Reducing frame resolution is a controllable way to reduce the detector input workload, but smaller "
        "faces may become harder to localize. This paper measures that trade-off for face detection only. "
        "It does not test identity recognition, crowded-scene load, or a complete surveillance pipeline.")
    add_body(template,
        "The contribution is a reproducible resolution benchmark for ChokePoint Portal 1 indoor frontal "
        "sequences. It uses sequence-disjoint training, development, and test sets; common eye-landmark-derived "
        "proxy boxes; fine-tuned YOLO and SCRFD detectors; a width and confidence sweep; and a synchronized "
        "throughput measurement on one laptop GPU. The benchmark is an empirical comparison under this "
        "protocol, not a new detector or a claim about either model family in general.")

    add_heading(template, "II. METHODOLOGY")
    add_heading(template, "A. Related Work", level=2)
    add_body(template,
        "YOLO established a single-stage formulation for real-time object detection [1]. This study uses "
        "Ultralytics implementations and their published nano checkpoints; the YOLOv5 initialization is the "
        "updated-head YOLOv5nu checkpoint, as distinguished in the current model documentation [5]; YOLOv12 "
        "is an attention-centric detector [6]. SCRFD "
        "redistributes samples and computation to target efficient face detection [4]. ChokePoint was collected "
        "for video-based face analysis at portal entrances and provides full-frame surveillance sequences and "
        "eye landmarks [2], [3].")

    add_heading(template, "B. Dataset and Split", level=2)
    add_body(template,
        "The source is the full-frame Portal 1 indoor frontal subset of ChokePoint. The split groups complete "
        "source sequences rather than randomly sampled frames: three sequences for training, one for "
        "development, and four for the locked test set. No sequence, dataset/frame pair, or adjacent-frame "
        "pair crosses splits. Session S2 and camera C2 occur in both training and development sequences; "
        "camera C1 occurs in training and test sequences. These metadata overlaps do not place "
        "a source sequence or frame in multiple splits. The split does not claim identity-disjoint subjects.")
    add_caption(template, "TABLE I. Sequence-level data partition.", table=True)
    sequence_rows = [
        ["Train", "P1E_S1_C1", "1,382"], ["Train", "P1L_S1_C1", "1,799"],
        ["Train", "P1L_S2_C2", "1,719"], ["Development", "P1E_S2_C2", "1,135"],
        ["Test", "P1E_S3_C3", "929"], ["Test", "P1E_S4_C1", "1,531"],
        ["Test", "P1L_S3_C3", "1,205"], ["Test", "P1L_S4_C1", "2,011"],
    ]
    add_table(template, ["Split", "Sequence", "Frames"], sequence_rows, [0.9, 1.55, 0.65])
    add_body(template,
        "The resulting totals are 4,900 training, 1,135 development, and 5,676 test frames. The test split "
        "was excluded from fitting, checkpoint selection, and operating-point selection. These counts refer "
        "to labeled full-frame instances available in the derived manifest; they do not represent all frames "
        "or all identities in the ChokePoint collection.")

    add_heading(template, "C. Eye-Landmark-Derived Boxes", level=2)
    add_body(template,
        "ChokePoint provides manually labeled eye landmarks rather than detector rectangles. For inter-eye "
        "distance d and eye midpoint (cx, cy), the proxy box is x1=cx−1.4d, x2=cx+1.4d, y1=cy−1.2d, and "
        "y2=cy+2.4d, clipped to the 800×600 frame. These deterministic boxes are not human-drawn face "
        "boundaries. Reported AP50, precision, recall, and F1 therefore measure agreement with this proxy "
        "label construction, not localization against manually annotated face rectangles.")
    add_body(template,
        "The latest user-designated hand-drawn file, boxes_only_240.csv, contains one box for each of 240 test frames, sampled evenly "
        "across four test sequences and three eye-distance strata. The file is used only as a secondary reference "
        "and does not change model or operating-point selection. Its coordinates exactly match the accompanying "
        "InsightFace proposal file, so this comparison is not an independent check against that model output. "
        "No second annotator or adjudication was recorded.")
    add_body(template,
        "To reduce manual review effort, a separate pseudo-reference set was formed from pretrained SCRFD-10GF "
        "and OpenCV Zoo YuNet proposals on the same 240 frames. When both selected boxes agreed at IoU 0.50 "
        "or higher and neither model returned multiple faces, their coordinates were averaged (164 frames). "
        "For the remaining 76 flagged frames, the largest YuNet proposal was retained and marked for review. "
        "These model-generated boxes are exploratory references, not human ground truth.")

    add_heading(template, "D. Detectors and Fine-Tuning", level=2)
    add_body(template,
        "The compared models are YOLOv5nu, YOLOv8n, and YOLOv12n initialized from local Ultralytics nano "
        "weights, and the SCRFD-500M face detector initialized from the checkpoint listed by the official "
        "InsightFace model zoo [4]. Each model was fine-tuned on the same 4,900 training instances and "
        "selected with the same 1,135 development instances. YOLO training used 20 epochs, 640-pixel "
        "training images, batch size 8, seed 391, and the recorded Ultralytics 8.4.54 defaults with the "
        "automatic optimizer selection. SCRFD used 20 epochs, batch size 2, seed 391, and its MMDetection "
        "training recipe with SGD at learning rate 0.0025. Recipe details differ by framework and are "
        "recorded in the run artifacts; the comparison controls the data split, not every optimizer or "
        "augmentation parameter.")
    add_body(template,
        "The SCRFD code is pinned to InsightFace commit c6c3af7 and uses a documented Windows compatibility "
        "patch for MMCV 1.7.2 and the installed Windows pycocotools package. The pretrained InsightFace "
        "model weights are distributed for non-commercial research; no commercial use is implied.")

    add_heading(template, "E. Metrics and Measurement Protocol", level=2)
    add_body(template,
        "The original aspect ratio is preserved at widths 640, 480, 384, 320, 288, 256, 240, 192, 160, and "
        "128 pixels. Confidence thresholds are 0.25, 0.35, 0.45, 0.55, 0.65, and 0.70; the IoU match "
        "threshold is 0.50. AP50 is calculated from detections retained above a 0.001 confidence floor "
        "using 101-point interpolated precision. At each listed threshold, a prediction is matched to an "
        "unmatched proxy box in the same frame; remaining detections and missed boxes contribute to FP and "
        "FN. The prespecified development rule requires AP50 and recall of at least 0.95 at confidence 0.25. "
        "It selects the smallest width satisfying both conditions; ties use higher development AP50, then "
        "recall, then the fixed model order. Test results do not change the selected model, width, or threshold.")
    add_body(template,
        "Throughput is measured on a fixed 600-frame sample, 150 frames from each test sequence, with 50 "
        "warm-up iterations and five full repeats. GPU work is synchronized around each timed repeat. The "
        "timed region includes image reading, framework resize/preprocessing, detector inference, and "
        "model postprocessing; model loading and metric calculation are excluded. We report median FPS and "
        "the interquartile range. This is detector throughput on the NVIDIA GeForce RTX 3050 Laptop GPU "
        "with 4 GB memory, not end-to-end recognition latency.")

    add_heading(template, "III. RESULTS")
    add_heading(template, "A. Full-Resolution Reference", level=2)
    full_rows = []
    for model in summary["model"].drop_duplicates().tolist():
        row = get_row(summary, model, 640, 0.25)
        full_rows.append([
            model, f"{row['ap50_vs_derived_boxes']:.3f}", f"{row['recall']:.3f}",
            f"{row['fps_median']:.1f}",
        ])
    add_caption(template, "TABLE II. Test results at 640-pixel width and confidence 0.25.", table=True)
    add_table(template, ["Model", "AP50", "Recall", "FPS"], full_rows, [1.3, 0.55, 0.6, 0.55])
    add_body(template,
        "Table II is a descriptive reference point on the locked test set. It does not determine which "
        "model or threshold is recommended. AP50 uses detections above the 0.001 confidence floor; precision, "
        "recall, and F1 use confidence 0.25. All values use the eye-landmark-derived proxy boxes.")

    add_heading(template, "B. Resolution and Confidence Trade-Off", level=2)
    if selection.get("selected_model"):
        model, width, confidence = selection["selected_model"], int(selection["selected_width"]), float(selection["confidence"])
        row = get_row(summary, model, width, confidence)
        add_body(template,
            f"Using development data only, the prespecified rule locked {model} at {width}×{round(width * 0.75)} pixels "
            f"and confidence {confidence:.2f}. Its test result was AP50 {row['ap50_vs_derived_boxes']:.3f}, "
            f"precision {row['precision']:.3f}, recall {row['recall']:.3f}, F1 {row['f1']:.3f}, and "
            f"median throughput {row['fps_median']:.1f} FPS (IQR {row['fps_q1']:.1f}–{row['fps_q3']:.1f}). "
            "This is a test estimate for the previously locked configuration, not a test-selected operating point.")
    else:
        add_body(template,
            "No model and resolution met both development-set thresholds at confidence 0.25, so no deployment "
            "operating point was selected. Test results are reported as a resolution and threshold sensitivity "
            "analysis only.")
    add_body(template,
        "Fig. 1 shows AP50 and recall across the width sweep at confidence 0.25. Fig. 2 reports throughput "
        "from the same fixed speed sample at each resolution. Fig. 3 shows precision, recall, and F1 across "
        "the confidence sweep for the development-locked setting. These curves support protocol-specific "
        "trade-off analysis; they do not establish that one architecture is generally superior.")
    for name, caption in [
        ("fig01_accuracy_by_resolution.png", "Fig. 1. Test AP50 and recall versus input width at confidence 0.25; labels are eye-landmark-derived proxy boxes."),
        ("fig02_throughput_by_resolution.png", "Fig. 2. Median detector throughput and interquartile range on the fixed 600-frame speed sample."),
    ]:
        paragraph = template.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(figures / name), width=Inches(3.15))
        add_caption(template, caption)
    if selection.get("selected_model"):
        paragraph = template.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(required_figures[2]), width=Inches(3.15))
        add_caption(template, "Fig. 3. Test threshold sensitivity for the model and resolution selected from development data.")
    else:
        paragraph = template.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(required_figures[2]), width=Inches(3.15))
        add_caption(template, "Fig. 3. Test F1 threshold sensitivity at 640-pixel width; no development candidate met the operating-point constraints.")

    add_heading(template, "C. Hand-Drawn Box Audit", level=2)
    audit_rows = []
    for model in summary["model"].drop_duplicates().tolist():
        row = get_manual_row(manual_summary, model, 288, 0.25)
        audit_rows.append([model, f"{row['ap50']:.3f}", f"{row['precision']:.3f}",
                           f"{row['recall']:.3f}", f"{row['f1']:.3f}"])
    add_caption(template, "TABLE III. Detector results against 240 hand-drawn test boxes at 288-pixel width. AP50 uses confidence floor 0.001; precision, recall, and F1 use 0.25.", table=True)
    add_table(template, ["Model", "AP50", "Precision", "Recall", "F1"], audit_rows,
              [1.35, 0.48, 0.68, 0.5, 0.42])
    agreement = manual_stats["manual_vs_proxy_iou"]
    add_body(template,
        f"The user-designated hand-drawn sample covers 240 frames. Mean IoU with landmark-derived proxies was "
        f"{agreement['mean']:.3f}; median IoU was {agreement['50%']:.3f} (IQR {agreement['25%']:.3f}–{agreement['75%']:.3f}), "
        f"with {int(round(manual_stats['manual_vs_proxy_iou_ge_0_5'] * 240))}/240 pairs at IoU 0.50 or higher. "
        "The boxes exactly match the accompanying InsightFace proposal coordinates, so the comparison is descriptive "
        "and does not independently validate those proposals. It was not used to retune or select the development-locked "
        "operating point. AP50 uses detections above the 0.001 confidence floor; the reported operating metrics use 0.25.")

    add_heading(template, "D. Model-Assisted Pseudo-Label Sensitivity", level=2)
    pseudo_rows = []
    for model in summary["model"].drop_duplicates().tolist():
        row = get_manual_row(pseudo_summary, model, 288, 0.25)
        pseudo_rows.append([model, f"{row['ap50']:.3f}", f"{row['precision']:.3f}",
                            f"{row['recall']:.3f}", f"{row['f1']:.3f}"])
    add_caption(template, "TABLE IV. Agreement with YuNet/SCRFD pseudo-boxes on 240 frames at 288-pixel width. AP50 uses confidence floor 0.001; precision, recall, and F1 use 0.25.", table=True)
    add_table(template, ["Model", "AP50", "Precision", "Recall", "F1"], pseudo_rows,
              [1.35, 0.48, 0.68, 0.5, 0.42])
    add_body(template,
        "These scores measure agreement with model-generated reference boxes and must not be read as human-label "
        "accuracy. The test models did not produce the reference boxes, but the references remain unverified; "
        "76 frames specifically require visual review.")

    add_heading(template, "IV. DISCUSSION AND CONCLUSION")
    add_heading(template, "A. Interpretation", level=2)
    add_body(template,
        "Resolution changes both detection quality and the detector workload under the stated transforms and "
        "hardware. The development-locked result provides one reproducible operating point when the preset "
        "AP50 and recall constraints can be met. The full test sweep is descriptive and should be interpreted "
        "against both the proxy-label definition and the 240-frame user-designated box audit, the model-specific "
        "training recipes, and the measured GPU only. "
        "An AP50 value near one would indicate close agreement with the derived boxes, not perfect face "
        "detection or recognition.")
    add_heading(template, "B. Limitations and Future Work", level=2)
    add_body(template,
        "The full resolution sweep is evaluated against eye-landmark-derived proxies; the separate user-designated "
        "hand-drawn audit covers 240 test frames from one annotator, has no inter-annotator agreement or adjudication, "
        "and its coordinates exactly match the accompanying InsightFace proposal file. The manual-reference comparison "
        "is therefore not independent of those proposals. The pseudo-label sensitivity check is model-generated "
        "and not human ground truth. "
        "The experiment covers only "
        "Portal 1 indoor frontal sequences and one RTX 3050 "
        "Laptop GPU; it does not establish cross-portal, outdoor, occlusion, identity-disjoint, or "
        "cross-hardware behavior. No downstream recognition accuracy, face-count stress test, or total "
        "pipeline latency was measured. Future work should add independent annotation and adjudication, evaluate "
        "additional portals and hardware, and measure the full detection-to-recognition pipeline.")
    add_heading(template, "C. Conclusion", level=2)
    add_body(template,
        "This work defines a sequence-disjoint benchmark for resolution-dependent face detection on "
        "ChokePoint Portal 1 indoor frontal video and compares fine-tuned YOLO and SCRFD models using "
        "the same train, development, and test instances. A development-only rule locks the candidate "
        "before test evaluation, while a separate fixed sample measures detector throughput. Conclusions "
        "remain limited to the derived-box full sweep, the 240-frame manual audit, listed sequences, and tested GPU; recognition "
        "performance and deployment safety require separate evaluation.")

    template.add_paragraph("REFERENCES", style="Heading 5")
    references = [
        "[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, real-time object detection,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition, 2016, pp. 779–788, doi: 10.1109/CVPR.2016.91.",
        "[2] Y. Wong, S. Chen, S. Mau, C. Sanderson, and B. C. Lovell, “Patch-based probabilistic image quality assessment for face selection and improved video-based face recognition,” in Proc. IEEE CVPR Workshops, 2011, pp. 81–88, doi: 10.1109/CVPRW.2011.5981881.",
        "[3] Y. Wong, S. Chen, S. Mau, C. Sanderson, and B. C. Lovell, “ChokePoint Dataset,” Zenodo, version 1, Jun. 22, 2017, doi: 10.5281/zenodo.815657.",
        "[4] J. Guo, J. Deng, A. Lattas, and S. Zafeiriou, “Sample and computation redistribution for efficient face detection,” in Proc. Int. Conf. Learning Representations, 2022. [Online]. Available: https://arxiv.org/abs/2105.04714.",
        "[5] Ultralytics, “Models supported by Ultralytics,” 2026. [Online]. Available: https://docs.ultralytics.com/models. Accessed: Nov. 14, 2026.",
        "[6] Y. Tian, Q. Ye, and D. Doermann, “YOLOv12: Attention-centric real-time object detectors,” in Advances in Neural Information Processing Systems, vol. 38, 2025, doi: 10.52202/085713-2627.",
    ]
    for reference in references:
        template.add_paragraph(reference, style="references")

    properties = template.core_properties
    properties.title = TITLE
    properties.author = "Vu Hai Minh; Cao Van Mai"
    properties.subject = "Detector-level resolution, accuracy, and throughput benchmark on ChokePoint Portal 1"
    properties.keywords = "face detection, ChokePoint, YOLO, SCRFD, resolution, throughput"

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    template.save(output)
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
