"""Fill the 2024 IEEE A4 template with the current paper text and figures."""
from __future__ import annotations

import re
import shutil
import sys
import os
from pathlib import Path

from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final"
sys.path.insert(0, str(FINAL))
import build_rewritten_paper as chart_builder  # charts are regenerated from summary.csv

# This is the namespace-compatible working copy of the 2024 IEEE A4 template.
TEMPLATE = ROOT / "docs" / "IEEE_Conference_Template_A4_2024_python_compatible.docx"
MARKDOWN = FINAL / "Paper_Rewritten_Final.md"
OUTPUT = FINAL / os.environ.get("PAPER_OUTPUT_NAME", "Paper_Rewritten_Final.docx")
FIGURE_DIR = FINAL / "figures" / "rewritten"

TITLE = "Minimum-Resolution Face Detection for Real-Time Surveillance: A Benchmark of YOLO and SCRFD"
def set_para_text(paragraph, text):
    """Replace paragraph text while retaining its paragraph and first-run properties."""
    runs = paragraph.runs
    if runs:
        runs[0].text = text
        for run in runs[1:]:
            run._element.getparent().remove(run._element)
    elif text:
        paragraph.add_run(text)


def remove_template_body(doc):
    body = doc._element.body
    elements = list(body.iterchildren())
    # Remove the floating graphic-instruction textbox shipped with the template.
    for el in list(elements):
        if el.tag == qn("w:p") and "We suggest that you use a text box" in "".join(el.itertext()):
            index = elements.index(el)
            if index and elements[index - 1].tag == qn("w:p"):
                previous = elements[index - 1]
                previous_properties = previous.find(qn("w:pPr"))
                has_section_break = (previous_properties is not None
                                     and previous_properties.find(qn("w:sectPr")) is not None)
                if not "".join(previous.itertext()) and not has_section_break:
                    body.remove(previous)
                    elements.remove(previous)
            body.remove(el)
            elements.remove(el)
    abstract_idx = next(i for i, el in enumerate(elements)
                        if el.tag == qn("w:p") and "Abstract" in "".join(el.itertext()))
    boundaries = [i for i, el in enumerate(elements)
                  if i > abstract_idx and el.tag == qn("w:p")
                  and el.find(qn("w:pPr")) is not None
                  and el.find(qn("w:pPr")).find(qn("w:sectPr")) is not None]
    boundary_idx = boundaries[-1]
    boundary = elements[boundary_idx]
    for el in elements[abstract_idx:boundary_idx]:
        body.remove(el)
    # Retain the template's section break and clear only its sample text.
    for child in list(boundary):
        if child.tag != qn("w:pPr"):
            boundary.remove(child)
    for el in list(body.iterchildren()):
        if el is boundary or el.tag != qn("w:p") or "".join(el.itertext()):
            continue
        ppr = el.find(qn("w:pPr"))
        if ppr is None or ppr.find(qn("w:sectPr")) is None:
            body.remove(el)
    return boundary


def add_paragraph_before(doc, boundary, text, style):
    paragraph = doc.add_paragraph(style=style)
    if text:
        paragraph.add_run(text)
    boundary.addprevious(paragraph._p)
    return paragraph


def set_author(paragraph, name, department, university, location, email):
    """Fill an existing template author slot; bold only the author's name."""
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)
    for index, value in enumerate((name, department, university, location, email)):
        run = paragraph.add_run()
        if index:
            run.add_break()
        run.add_text(value)
        if index == 0:
            run.bold = True


def add_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        border = borders.find(tag)
        if border is None:
            border = OxmlElement(f"w:{edge}")
            borders.append(border)
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "8" if edge in ("top", "bottom") else "4")
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), "7F8C99")


def make_author_table(doc, first_author, second_author):
    """Keep both authors top-aligned across the full-width title section."""
    table = doc.add_table(rows=1, cols=2)
    table.style = "Normal Table"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    width = doc.sections[1].page_width - doc.sections[1].left_margin - doc.sections[1].right_margin
    half_width = Inches(width / 914400 / 2)
    first_author._p.addprevious(table._tbl)
    for i, (cell, author) in enumerate(zip(table.rows[0].cells, (first_author, second_author))):
        cell.width = half_width
        table.columns[i].width = half_width
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
        cell._tc.remove(cell.paragraphs[0]._p)
        author.alignment = WD_ALIGN_PARAGRAPH.CENTER
        author.paragraph_format.keep_together = True
        cell._tc.append(author._p)

    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "nil")
        borders.append(border)
    table._tbl.tblPr.append(borders)


def add_table_before(doc, boundary, rows):
    ncols = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.style = "Normal Table"
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    add_table_borders(table)
    widths = table_widths(rows)
    for col, width in zip(table.columns, widths):
        col.width = Inches(width)
    for ri, values in enumerate(rows):
        for ci, value in enumerate(values):
            cell = table.cell(ri, ci)
            cell.width = Inches(widths[ci])
            p = cell.paragraphs[0]
            p.style = "table col head" if ri == 0 else "table copy"
            p.add_run(value)
            if ri == 0:
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "E6EDF5")
                cell._tc.get_or_add_tcPr().append(shading)
    boundary.addprevious(table._tbl)


def table_widths(rows):
    """Column widths in inches, sized to the template's two-column text width."""
    n = len(rows[0])
    if n == 3:
        return [0.72, 1.83, 0.65]
    if n == 4:
        return [1.22, 0.64, 0.64, 0.70]
    if rows[0][1].lower().startswith("precision"):
        return [1.05, 0.55, 0.62, 0.52, 0.50]
    return [1.05, 0.58, 0.52, 0.55, 0.54]


def is_separator(line):
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_table(lines):
    rows = []
    for line in lines:
        if is_separator(line):
            continue
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def add_image_before(doc, boundary, markdown_line):
    match = re.search(r"figures/rewritten/([^\)]+)", markdown_line)
    if not match:
        return
    image_path = FIGURE_DIR / match.group(1)
    paragraph = doc.add_paragraph(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(image_path), width=Inches(3.15))
    boundary.addprevious(paragraph._p)


def insert_manuscript(doc, boundary):
    lines = MARKDOWN.read_text(encoding="utf-8").splitlines()
    # The title and two-author Markdown table are represented by template author slots.
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith("# ")):
        i += 1
    if i < len(lines) and "Vu Hai Minh" in lines[i]:
        i += 1
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            i += 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("![Figure"):
            add_image_before(doc, boundary, line)
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = parse_table(table_lines)
            if rows and rows[0] and rows[0][0] not in {"Vu Hai Minh / Department of Artificial Intelligence / FPT University / Ha Noi, Viet Nam / vhaiminh.2k5@gmail.com"}:
                add_table_before(doc, boundary, rows)
            continue
        if line.startswith("## "):
            heading = line[3:].strip()
            if heading == "REFERENCES":
                style = "Heading 5"
            elif re.match(r"^[IVX]+\.", heading):
                style = "Heading 1"
                heading = re.sub(r"^[IVX]+\.\s*", "", heading)
            else:
                style = "Heading 2"
                heading = re.sub(r"^[A-Z]\.\s*", "", heading)
            add_paragraph_before(doc, boundary, heading, style)
            i += 1
            continue
        if line.startswith("# "):
            i += 1
            continue
        if line.startswith("Abstract—"):
            style = "Abstract"
        elif line.startswith("Index Terms—"):
            style = "Keywords"
        elif line.startswith("TABLE "):
            style = "table head"
            line = re.sub(r"^TABLE\s+[IVX]+\.\s*", "", line)
        elif line.startswith("Fig. "):
            style = "figure caption"
            line = re.sub(r"^Fig\.\s*\d+\.\s*", "", line)
        elif re.match(r"^\[\d+\]", line):
            style = "references"
        else:
            style = "Body Text"
        paragraph = add_paragraph_before(doc, boundary, line, style)
        if style == "table head":
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.keep_together = True
        i += 1


def main():
    # Regenerate all charts from the canonical benchmark before embedding them.
    chart_builder.draw_accuracy()
    chart_builder.draw_throughput()
    chart_builder.draw_threshold_sensitivity()
    chart_builder.draw_f1_sensitivity_by_model()

    shutil.copy2(TEMPLATE, OUTPUT)
    doc = Document(OUTPUT)
    paragraphs = doc.paragraphs
    set_para_text(paragraphs[0], TITLE)
    body = doc._element.body
    for paragraph in (paragraphs[1], paragraphs[2]):
        body.remove(paragraph._p)
    paragraphs = doc.paragraphs
    set_para_text(paragraphs[1], "")
    set_author(paragraphs[2], "Vu Hai Minh", "Department of Artificial Intelligence",
               "FPT University", "Ha Noi, Viet Nam", "vhaiminh.2k5@gmail.com")
    set_author(paragraphs[3], "Cao Van Mai", "Department of Artificial Intelligence",
               "FPT University", "Ha Noi, Viet Nam", "maicv2@fe.edu.vn")
    make_author_table(doc, paragraphs[2], paragraphs[3])

    # The template reserves three author columns. Keep the author section's
    # margins and type sizes, use two columns, and remove its spare slots/break.
    for paragraph in (paragraphs[4], paragraphs[5], paragraphs[7]):
        if paragraph._p.getparent() is body:
            body.remove(paragraph._p)
    for el in body.iterchildren():
        if el.tag != qn("w:p"):
            continue
        ppr = el.find(qn("w:pPr"))
        sect = ppr.find(qn("w:sectPr")) if ppr is not None else None
        cols = sect.find(qn("w:cols")) if sect is not None else None
        if cols is not None and cols.get(qn("w:num")) == "3":
            cols.set(qn("w:num"), "1")
            break

    boundary = remove_template_body(doc)
    insert_manuscript(doc, boundary)
    doc.core_properties.title = TITLE
    doc.core_properties.subject = "Accuracy-throughput benchmark for fine-tuned YOLO and SCRFD detectors"
    doc.save(OUTPUT)
    remove_ieee_footer_placeholder(OUTPUT)
    print(OUTPUT)


def remove_ieee_footer_placeholder(path):
    """Clear the example IEEE copyright string while retaining footer layout."""
    from tempfile import NamedTemporaryFile
    from zipfile import ZIP_DEFLATED, ZipFile
    from lxml import etree

    with NamedTemporaryFile(dir=path.parent, suffix=".docx", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        with ZipFile(path, "r") as src, ZipFile(temp_path, "w", ZIP_DEFLATED) as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename.startswith("word/footer") and item.filename.endswith(".xml"):
                    root = etree.fromstring(data)
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    for paragraph in root.xpath('.//w:p[.//w:t[contains(., "XXX-X-XXXX")]]', namespaces=ns):
                        for child in list(paragraph):
                            if child.tag != "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr":
                                paragraph.remove(child)
                    data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
                dst.writestr(item, data)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    main()
