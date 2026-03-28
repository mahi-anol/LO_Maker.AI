"""
DOCX generator: fills the actual lesson plan template (template.docx)
with AI-generated Bangla content.

Strategy:
  - Copy template.docx to output path
  - Clear only the dynamic content cells
  - Write new Bangla content into those cells
  - Keep ALL fixed English text, colors, borders untouched

Cell map (from template analysis):
  TABLE 0 — Header info
    R0C1 → teacher_name
    R0C3 → subject
    R1C1 → grade
    R1C3 → duration

  TABLE 1 — Lesson Vision
    R2C0  → learning_outcome
    R4C0  → assessment questions + time/marks
    R6C0  → exemplar answer
    R8C0  → what + why (academic) + why (non-academic)
    R8C1  → how (key points steps)
    R10C0 → knowledge from LO
    R12C0 → bloom's verb / skill

  TABLE 2 — Lesson Method (content rows only, fixed rows untouched)
    R3C0  → launch teacher action
    R3C1  → launch student action
    R3C2  → launch materials
    R6C0  → explore teacher action
    R6C1  → explore student action
    R6C2  → explore materials
    R9C0  → conceptualize teacher action
    R9C1  → conceptualize student action
    R9C2  → conceptualize materials
    R12C0 → guided practice teacher action
    R12C1 → guided practice student action
    R12C2 → guided practice materials
    R15C0 → independent practice teacher action
    R15C1 → independent practice student action
    R15C2 → independent practice materials
    R18C0 → lesson closing teacher action
    R18C1 → lesson closing student action
    R18C2 → lesson closing materials
"""

import os
import shutil
import copy
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template.docx")

# ── Parse ### sections from LLM output ────────────────────────────────────────

def parse_sections(text: str) -> dict:
    result = {}
    current_key = "__default__"
    current_lines = []
    for line in text.splitlines():
        if line.startswith("###"):
            if current_lines:
                result[current_key] = "\n".join(current_lines).strip()
            current_key = line.replace("###", "").strip().rstrip(":")
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        result[current_key] = "\n".join(current_lines).strip()
    return result


# ── Cell writing helpers ───────────────────────────────────────────────────────

def get_cell_font_info(cell):
    """Extract font name and size from first run of cell (to preserve styling)."""
    font_name = "Noto Sans Bengali"
    font_size = None
    bold = False
    for para in cell.paragraphs:
        for run in para.runs:
            if run.font.name:
                font_name = run.font.name
            if run.font.size:
                font_size = run.font.size
            if run.bold:
                bold = run.bold
            break
        break
    return font_name, font_size, bold


def clear_cell(cell):
    """Remove all paragraphs from a cell, leaving one empty paragraph."""
    # Keep the first paragraph, remove the rest
    tc = cell._tc
    paras = tc.findall(qn("w:p"))
    # Remove all but first
    for para in paras[1:]:
        tc.remove(para)
    # Clear the first paragraph's runs
    first_para = paras[0] if paras else OxmlElement("w:p")
    for r in first_para.findall(qn("w:r")):
        first_para.remove(r)
    for r in first_para.findall(qn("w:hyperlink")):
        first_para.remove(r)


def write_cell(cell, text: str, font_name="Noto Sans Bengali",
               font_size=None, bold=False, preserve_first_para_fmt=True):
    """
    Write multi-line text into a cell.
    Preserves cell background, borders, and paragraph formatting.
    Each line in text becomes a separate paragraph.
    """
    from docx.shared import Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from lxml import etree

    tc = cell._tc

    # Get existing paragraph properties from first paragraph (for formatting reference)
    existing_paras = tc.findall(qn("w:p"))
    first_pPr = None
    first_rPr = None
    if existing_paras:
        p0 = existing_paras[0]
        first_pPr = p0.find(qn("w:pPr"))
        for r in p0.findall(qn("w:r")):
            first_rPr = r.find(qn("w:rPr"))
            break

    # Clear existing content
    clear_cell(cell)

    lines = text.splitlines()
    if not lines:
        lines = [""]

    existing_paras_after = tc.findall(qn("w:p"))

    for i, line in enumerate(lines):
        if i == 0 and existing_paras_after:
            # Reuse the existing first paragraph
            para_el = existing_paras_after[0]
        else:
            # Create a new paragraph element
            para_el = OxmlElement("w:p")
            # Copy paragraph properties if available
            if first_pPr is not None:
                new_pPr = copy.deepcopy(first_pPr)
                para_el.insert(0, new_pPr)
            tc.append(para_el)

        # Create a run
        run_el = OxmlElement("w:r")

        # Build rPr
        rPr = OxmlElement("w:rPr")

        # Font
        rFonts = OxmlElement("w:rFonts")
        rFonts.set(qn("w:ascii"), font_name)
        rFonts.set(qn("w:hAnsi"), font_name)
        rFonts.set(qn("w:cs"), font_name)
        rPr.append(rFonts)

        # Size
        if font_size:
            sz = OxmlElement("w:sz")
            sz.set(qn("w:val"), str(int(font_size.pt * 2)))
            rPr.append(sz)
            szCs = OxmlElement("w:szCs")
            szCs.set(qn("w:val"), str(int(font_size.pt * 2)))
            rPr.append(szCs)

        # Bold
        if bold:
            b = OxmlElement("w:b")
            rPr.append(b)
            bCs = OxmlElement("w:bCs")
            rPr.append(bCs)

        # Preserve existing rPr color/style if available
        if first_rPr is not None:
            for child in first_rPr:
                tag = child.tag
                # Don't duplicate font/size/bold we already set
                skip_tags = [qn("w:rFonts"), qn("w:sz"), qn("w:szCs"),
                             qn("w:b"), qn("w:bCs")]
                if tag not in skip_tags:
                    rPr.append(copy.deepcopy(child))

        run_el.append(rPr)

        # Text
        t_el = OxmlElement("w:t")
        t_el.text = line if line else " "
        if line != line.strip() or not line:
            t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        run_el.append(t_el)

        para_el.append(run_el)


# ── Content assemblers ─────────────────────────────────────────────────────────

def build_assessment_text(assessment: str) -> tuple:
    """Returns (questions_text, exemplar_text, time_marks_text)"""
    secs = parse_sections(assessment)
    time_marks = secs.get("সময় ও পূর্ণমান", "সময়: ৪ মিনিট | পূর্ণমান: ৬ মার্ক")
    questions  = secs.get("প্রশ্নসমূহ", assessment)
    exemplar   = secs.get("আদর্শ উত্তর (Exemplar)", "")
    return questions, exemplar, time_marks


def build_vision_texts(lesson_plan: dict) -> tuple:
    """Returns (what_why_text, how_text)"""
    vision_secs = parse_sections(lesson_plan.get("lesson_vision", ""))
    what   = vision_secs.get("WHAT (Concept)", vision_secs.get("WHAT", ""))
    why_ac = vision_secs.get("WHY (Academic)", "")
    why_no = vision_secs.get("WHY (Non-Academic)", "")

    what_why_lines = []
    if what:
        what_why_lines.append("What:")
        what_why_lines.extend(what.splitlines())
        what_why_lines.append("")
    if why_ac:
        what_why_lines.append("Why (Academic):")
        what_why_lines.extend(why_ac.splitlines())
        what_why_lines.append("")
    if why_no:
        what_why_lines.append("Why (Non-Academic):")
        what_why_lines.extend(why_no.splitlines())

    how_text = lesson_plan.get("key_points", "")
    return "\n".join(what_why_lines), how_text


def build_method_section(content: str) -> tuple:
    """
    Returns (teacher_action, student_action, materials)
    Parses LLM output sections into the 3 columns.
    """
    secs = parse_sections(content)

    # Teacher action = everything except student action / materials
    teacher_lines = []
    student_lines = []
    materials_lines = []

    student_keys  = {"Student Action", "শিক্ষার্থীর কাজ", "শিক্ষার্থীরা"}
    material_keys = {"Materials", "উপকরণ", "Materials needed"}

    for key, val in secs.items():
        if key == "__default__":
            teacher_lines.extend(val.splitlines())
        elif any(sk in key for sk in student_keys):
            student_lines.extend(val.splitlines())
        elif any(mk in key for mk in material_keys):
            materials_lines.extend(val.splitlines())
        else:
            # Put into teacher action with label
            teacher_lines.append(f"【{key}】")
            teacher_lines.extend(val.splitlines())
            teacher_lines.append("")

    # Fallback: if nothing parsed, use full content as teacher action
    if not teacher_lines and not secs:
        teacher_lines = content.splitlines()

    teacher_action  = "\n".join(teacher_lines).strip()
    student_action  = "\n".join(student_lines).strip() if student_lines else "শিক্ষার্থীরা মনোযোগ দিয়ে অংশগ্রহণ করে।"
    materials       = "\n".join(materials_lines).strip() if materials_lines else "খাতা, কলম"

    return teacher_action, student_action, materials


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_docx(lesson_plan: dict, output_path: str) -> str:
    """
    Fill the template.docx with lesson_plan content and save to output_path.
    All fixed English content, colors, borders remain exactly as in the template.
    """
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(
            f"Template not found at {TEMPLATE_PATH}. "
            "Please place template.docx in the same folder as docx_generator.py"
        )

    # Work on a copy
    shutil.copy2(TEMPLATE_PATH, output_path)
    doc = Document(output_path)

    t0 = doc.tables[0]  # Header
    t1 = doc.tables[1]  # Lesson Vision
    t2 = doc.tables[2]  # Lesson Method

    # ── TABLE 0: Header ────────────────────────────────────────────────────────
    write_cell(t0.rows[0].cells[1], lesson_plan.get("teacher_name", ""))
    write_cell(t0.rows[0].cells[3], lesson_plan.get("subject", ""))
    write_cell(t0.rows[1].cells[1], lesson_plan.get("grade", ""))
    write_cell(t0.rows[1].cells[3], lesson_plan.get("duration", ""))

    # ── TABLE 1: Lesson Vision ─────────────────────────────────────────────────

    # R2C0 → Learning Outcome
    write_cell(t1.rows[2].cells[0], lesson_plan.get("learning_outcome", ""))

    # R4C0 → Assessment (time + questions)
    questions, exemplar, time_marks = build_assessment_text(
        lesson_plan.get("assessment", "")
    )
    assess_text = f"{time_marks}\n\n{questions}"
    write_cell(t1.rows[4].cells[0], assess_text)

    # R6C0 → Exemplar
    write_cell(t1.rows[6].cells[0], exemplar)

    # R8C0 → What + Why | R8C1 → How
    what_why_text, how_text = build_vision_texts(lesson_plan)
    write_cell(t1.rows[8].cells[0], what_why_text)
    write_cell(t1.rows[8].cells[1], how_text)

    # R10C0 → Knowledge from LO (keep fixed label in R9, fill content in R10)
    knowledge_text = lesson_plan.get("learning_outcome", "")  # brief reuse
    # Use lesson_vision what section for knowledge
    vision_secs = parse_sections(lesson_plan.get("lesson_vision", ""))
    what_text = vision_secs.get("WHAT (Concept)", vision_secs.get("WHAT", ""))
    write_cell(t1.rows[10].cells[0], what_text[:200] if what_text else "")

    # R12C0 → Bloom's skill
    key_points = lesson_plan.get("key_points", "")
    # Extract first line as the skill descriptor
    skill_line = key_points.splitlines()[0] if key_points else ""
    write_cell(t1.rows[12].cells[0], skill_line)

    # ── TABLE 2: Lesson Method ─────────────────────────────────────────────────
    # Fixed rows: 0,1,2,4,5,7,8,10,11,13,14,16,17 → DO NOT TOUCH
    # Dynamic rows: 3, 6, 9, 12, 15, 18

    sections_map = [
        (3,  "launch"),
        (6,  "explore"),
        (9,  "conceptualize"),
        (12, "guided_practice"),
        (15, "independent_practice"),
        (18, "lesson_closing"),
    ]

    for row_idx, key in sections_map:
        content = lesson_plan.get(key, "")
        teacher_action, student_action, materials = build_method_section(content)

        row = t2.rows[row_idx]
        write_cell(row.cells[0], teacher_action)
        write_cell(row.cells[1], student_action)
        write_cell(row.cells[2], materials)

    doc.save(output_path)
    return output_path