"""
PDF generator: converts the filled template DOCX to PDF.

Strategy: fill the template DOCX first (via docx_generator),
then convert it to PDF using LibreOffice headless.
This perfectly preserves all colors, fonts, borders, and layout.

Fallback: if LibreOffice is not available, uses docx2pdf (Windows/Mac).
"""

import os
import shutil
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)


def _libreoffice_convert(docx_path: str, output_dir: str) -> str:
    """Convert DOCX to PDF using LibreOffice headless. Returns PDF path."""
    # Try common LibreOffice executable names
    lo_executables = [
        "libreoffice",
        "soffice",
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        "/usr/bin/libreoffice",
        "/usr/bin/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]

    lo_exe = None
    for exe in lo_executables:
        if shutil.which(exe) or os.path.exists(exe):
            lo_exe = exe
            break

    if not lo_exe:
        raise FileNotFoundError("LibreOffice not found.")

    cmd = [
        lo_exe,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", output_dir,
        docx_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")

    # LibreOffice names the output same as input with .pdf extension
    base = os.path.splitext(os.path.basename(docx_path))[0]
    pdf_path = os.path.join(output_dir, base + ".pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Expected PDF not found at {pdf_path}")
    return pdf_path


def _docx2pdf_convert(docx_path: str, pdf_path: str) -> str:
    """Convert using docx2pdf (works on Windows with Word, or Mac)."""
    try:
        from docx2pdf import convert
        convert(docx_path, pdf_path)
        return pdf_path
    except ImportError:
        raise ImportError("docx2pdf not installed. Run: pip install docx2pdf")


def generate_pdf(lesson_plan: dict, output_path: str) -> str:
    """
    Generate PDF from lesson_plan dict.
    First fills the DOCX template, then converts to PDF.
    Returns output_path on success.
    """
    from docx_generator import generate_docx

    # Step 1: Fill the DOCX template
    tmp_dir = tempfile.mkdtemp()
    tmp_docx = os.path.join(tmp_dir, "lesson_plan_tmp.docx")
    generate_docx(lesson_plan, tmp_docx)
    logger.info(f"DOCX filled: {tmp_docx}")

    # Step 2: Convert to PDF
    pdf_output_dir = os.path.dirname(output_path)

    # Try LibreOffice first (works on Linux/HuggingFace and most systems)
    try:
        tmp_pdf = _libreoffice_convert(tmp_docx, tmp_dir)
        shutil.copy2(tmp_pdf, output_path)
        logger.info(f"PDF generated via LibreOffice: {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"LibreOffice conversion failed: {e}. Trying docx2pdf...")

    # Try docx2pdf (Windows with Word installed, or Mac)
    try:
        _docx2pdf_convert(tmp_docx, output_path)
        logger.info(f"PDF generated via docx2pdf: {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"docx2pdf failed: {e}. Falling back to basic PDF...")

    # Final fallback: reportlab basic PDF with note to user
    try:
        _fallback_reportlab(lesson_plan, output_path)
        logger.info(f"PDF generated via reportlab fallback: {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(
            f"All PDF generation methods failed. "
            f"Please install LibreOffice for best results. Error: {e}"
        )


def _fallback_reportlab(lesson_plan: dict, output_path: str):
    """
    Basic fallback PDF using reportlab.
    Used only when LibreOffice and docx2pdf are unavailable.
    Note: this does NOT match the template visually — install LibreOffice for
    a pixel-perfect PDF.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    BLUE = colors.HexColor("#CFE2F3")
    DARK = colors.HexColor("#D9D9D9")

    # Try to register Bengali font
    font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    font_reg = "Helvetica"
    for fn in ["NotoSansBengali-Regular.ttf", "DejaVuSans.ttf"]:
        fp = os.path.join(font_dir, fn)
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont("BengFont", fp))
                font_reg = "BengFont"
                break
            except Exception:
                pass

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("body", fontName=font_reg, fontSize=9,
                                leading=13, wordWrap="RTL")
    header_style = ParagraphStyle("hdr", fontName=font_reg, fontSize=10,
                                  leading=14, textColor=colors.black)

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []

    def add_row(label, value):
        t = Table([[Paragraph(label, header_style), Paragraph(value, body_style)]],
                  colWidths=[4*cm, 13*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(0,0), DARK),
            ("GRID", (0,0),(-1,-1), 0.5, colors.grey),
            ("TOPPADDING", (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING", (0,0),(-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 4))

    def add_section(title, content):
        t = Table([[Paragraph(title, header_style)]],
                  colWidths=[17*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), BLUE),
            ("GRID", (0,0),(-1,-1), 0.5, colors.grey),
            ("TOPPADDING", (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING", (0,0),(-1,-1), 8),
        ]))
        story.append(t)
        for line in content.splitlines():
            if line.strip():
                story.append(Paragraph(line, body_style))
        story.append(Spacer(1, 8))

    add_row("শিক্ষকের নাম:", lesson_plan.get("teacher_name",""))
    add_row("বিষয়:", lesson_plan.get("subject",""))
    add_row("শ্রেণি:", lesson_plan.get("grade",""))
    add_row("সময়:", lesson_plan.get("duration",""))
    story.append(Spacer(1, 10))

    add_section("শিক্ষার ফলাফল", lesson_plan.get("learning_outcome",""))
    add_section("Lesson Vision", lesson_plan.get("lesson_vision",""))
    add_section("Key Points", lesson_plan.get("key_points",""))
    add_section("Assessment", lesson_plan.get("assessment",""))
    add_section("Launch", lesson_plan.get("launch",""))
    add_section("Explore", lesson_plan.get("explore",""))
    add_section("Conceptualize", lesson_plan.get("conceptualize",""))
    add_section("Guided Practice", lesson_plan.get("guided_practice",""))
    add_section("Independent Practice", lesson_plan.get("independent_practice",""))
    add_section("Lesson Closing", lesson_plan.get("lesson_closing",""))

    story.append(Paragraph(
        "⚠️ Note: Install LibreOffice for a PDF that matches the template exactly.",
        body_style
    ))

    doc.build(story)