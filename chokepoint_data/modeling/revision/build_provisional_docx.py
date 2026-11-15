from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, RGBColor

from build_final_docx import (
    TITLE,
    add_body,
    add_caption,
    add_heading,
    add_table,
    clear_body_preserve_final_section,
    set_columns,
    set_table_borders,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an explicitly provisional IEEE A4 review draft from dev-only YOLO results.")
    parser.add_argument("--template", default="docs/IEEE_Conference_Template_A4_2024_python_compatible.docx")
    parser.add_argument("--summary", default="chokepoint_data/modeling/final_revision/benchmark_dev_provisional/dev_summary.csv")
    parser.add_argument("--selection", default="chokepoint_data/modeling/final_revision/benchmark_dev_provisional/operating_point_dev_locked.json")
    parser.add_argument("--out", default="docs/Paper_Improved_Provisional.docx")
    parser.add_argument("--figure", default="chokepoint_data/modeling/final_revision/figures/provisional_dev_only_ap50_recall.png")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    if summary.empty or not summary["split"].eq("dev").all():
        raise ValueError("Provisional draft accepts development rows only")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if not selection.get("provisional"):
        raise ValueError("The development selection must be explicitly marked provisional")

    figure_path = Path(args.figure)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.15, 2.25))
    for model, group in summary[summary["conf_threshold"] == 0.25].groupby("model", sort=False):
        group = group.sort_values("resolution_width", ascending=False)
        ax.plot(group["resolution_width"], group["ap50_vs_derived_boxes"], marker="o", label=f"{model} AP50")
        ax.plot(group["resolution_width"], group["recall"], marker="s", linestyle="--", label=f"{model} recall")
    ax.set_xlabel("Input width (pixels)", fontsize=7)
    ax.set_ylabel("Dev score", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_ylim(0.0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=5.2, ncol=2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    template = Document(args.template)
    clear_body_preserve_final_section(template)
    section = template.sections[0]
    section.page_width = Inches(8.268)
    section.page_height = Inches(11.693)
    set_columns(section, 1, 36)

    title = template.add_paragraph(TITLE, style="paper title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    authors = template.add_table(rows=1, cols=2)
    authors.autofit = False
    set_table_borders(authors, style="none")
    author_info = [
        ["Vu Hai Minh", "Department of Artificial Intelligence", "FPT University", "Ha Noi, Viet Nam", "vhaiminh.2k5@gmail.com"],
        ["Cao Van Mai", "Department of Artificial Intelligence", "FPT University", "Ha Noi, Viet Nam", "maicv2@fe.edu.vn"],
    ]
    for cell, lines in zip(authors.rows[0].cells, author_info):
        cell.text = ""
        for index, line in enumerate(lines):
            paragraph = cell.paragraphs[0] if index == 0 else cell.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.style = template.styles["Author"]
            paragraph.add_run(line)
    notice = template.add_paragraph()
    notice.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = notice.add_run("PROVISIONAL REVIEW DRAFT - NOT READY FOR SUBMISSION")
    run.bold = True
    run.font.color.rgb = RGBColor(170, 0, 0)

    body_section = template.add_section(WD_SECTION.CONTINUOUS)
    set_columns(body_section, 2, 18)
    abstract = template.add_paragraph(style="Abstract")
    abstract.add_run("Abstract— ").bold = True
    abstract.add_run(
        "This provisional draft reports development-only results for three fine-tuned YOLO nano models on "
        "ChokePoint Portal 1 indoor frontal video. The 4,900/1,135/5,676 sequence split and the ten-width "
        "resolution sweep are prepared; YOLOv5nu, YOLOv8n, and YOLOv12n completed 20 training epochs. "
        f"Among these three models, the development rule provisionally selected {selection['selected_model']} "
        f"at {selection['selected_width']} pixels and confidence 0.25 (AP50 {selection['dev_ap50_vs_derived_boxes']:.3f}, "
        f"recall {selection['dev_recall']:.3f}). SCRFD fine-tuning and all test evaluation remain pending, "
        "so this selection is not a final operating point. Labels are eye-landmark-derived proxy boxes, and "
        "the planned human box audit is also pending. No throughput or recognition result is reported."
    )
    keywords = template.add_paragraph(style="Keywords")
    keywords.add_run("Index Terms— ").bold = True
    keywords.add_run("face detection, ChokePoint, resolution, development set, proxy labels, YOLO.")

    add_heading(template, "I. INTRODUCTION")
    add_body(template,
        "Input resolution changes the visual detail and computation presented to a face detector. This work "
        "is designed to measure that trade-off for Portal 1 indoor frontal sequences in ChokePoint. The scope "
        "is detector-level only; it does not measure identity recognition or an end-to-end surveillance pipeline.")
    add_body(template,
        "This file is an internal, provisional draft. It preserves the submitted paper title and an editable "
        "author block copied from the submitted PDF. The author details are a draft for the user to correct.")

    add_heading(template, "II. DATA AND EXPERIMENT DESIGN")
    add_heading(template, "A. Split and labels", level=2)
    add_body(template,
        "Complete source sequences were assigned to 4,900 training, 1,135 development, and 5,676 test frames. "
        "The split audit found no sequence, dataset/frame, duplicate-path, or adjacent-frame crossing. Face-box "
        "targets are derived from eye landmarks and are proxies rather than human-drawn face rectangles. "
        "A blinded 240-frame manual box audit from a 360-frame candidate pool has not been annotated.")
    add_heading(template, "B. Models and training", level=2)
    add_body(template,
        "YOLOv5nu, YOLOv8n, and YOLOv12n were fine-tuned for 20 epochs from locally verified pretrained "
        "weights on the shared training split, with model selection restricted to development data. SCRFD-500M "
        "was not fine-tuned: its Windows training path requires an MSVC C++ compiler for a PyTorch/MMCV "
        "extension, and installation of Microsoft Build Tools was canceled during UAC elevation. No SCRFD "
        "result is included in this draft.")
    add_heading(template, "C. Provisional development protocol", level=2)
    add_body(template,
        "The development sweep covered nominal widths 640, 480, 384, 320, 288, 256, 240, 192, 160, and 128 "
        "pixels, with confidence 0.25 for the provisional selection rule and IoU 0.50. The rule requires "
        "AP50 of at least 0.95 and recall of at least 0.95, then prefers the smallest eligible width. The "
        "development results below include only the three YOLO models. Test labels and images were not loaded.")

    add_heading(template, "III. PROVISIONAL DEVELOPMENT RESULTS")
    model_names = summary["model"].drop_duplicates().tolist()
    rows = []
    for model in model_names:
        row640 = summary[(summary.model == model) & (summary.resolution_width == 640) & (summary.conf_threshold == 0.25)].iloc[0]
        rows.append([model, "640", f"{row640.ap50_vs_derived_boxes:.3f}", f"{row640.precision:.3f}", f"{row640.recall:.3f}"])
    selected = summary[
        (summary.model == selection["selected_model"])
        & (summary.resolution_width == int(selection["selected_width"]))
        & (summary.conf_threshold == 0.25)
    ].iloc[0]
    rows.append([selection["selected_model"], str(selection["selected_width"]),
                 f"{selected.ap50_vs_derived_boxes:.3f}", f"{selected.precision:.3f}", f"{selected.recall:.3f}"])
    add_table(template, ["Model", "Width", "AP50", "Precision", "Recall"], rows,
              [1.3, 0.65, 0.7, 0.8, 0.7])
    paragraph = template.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(figure_path), width=Inches(3.15))
    add_caption(template, "Fig. 1. Provisional development AP50 and recall at confidence 0.25 against landmark-derived proxy boxes.")
    add_body(template,
        f"Among the three evaluated YOLO models, the provisional development rule selected {selection['selected_model']} "
        f"at {selection['selected_width']} pixels (AP50 {selection['dev_ap50_vs_derived_boxes']:.3f}, "
        f"precision {selection['dev_precision']:.3f}, recall {selection['dev_recall']:.3f}, F1 {selection['dev_f1']:.3f}). "
        "This is a development result on proxy labels, not a test estimate or a final recommendation. The SCRFD model "
        "could change the full-model operating point, so no final model ranking is claimed.")

    add_heading(template, "IV. REMAINING WORK AND LIMITATIONS")
    add_body(template,
        "The final comparison requires SCRFD fine-tuning on the same train/development data, a fresh development "
        "selection across all four models, and then the locked test sweep. The 600-frame speed protocol has not run. "
        "The manual box audit, human occlusion assessment, and inter-annotator review also remain pending. Results "
        "cannot yet support test-set accuracy, throughput, human-box agreement, generalization beyond Portal 1, "
        "or downstream recognition claims. Word rendering and visual inspection are not complete.")

    template.add_paragraph("REFERENCES", style="Heading 5")
    for reference in [
        "[1] Y. Wang et al., ‘YOLOv12: Attention-centric real-time object detectors,’ NeurIPS, 2025.",
        "[2] Y. Wu, M. Mahmoudi, and S. S. Cheung, ‘ChokePoint: A video dataset for face identification in a doorway,’ IEEE BTAS, 2011.",
        "[3] J. Guo et al., ‘Sample and computation redistribution for efficient face detection,’ ICLR, 2022.",
    ]:
        template.add_paragraph(reference, style="references")

    properties = template.core_properties
    properties.title = TITLE + " (Provisional Dev-Only Draft)"
    properties.author = "Vu Hai Minh; Cao Van Mai"
    properties.subject = "Provisional development-only review draft; no test results"
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    template.save(output)
    print(f"Wrote provisional draft {output.resolve()}")


if __name__ == "__main__":
    main()
