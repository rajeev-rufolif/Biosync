"""
document_parser.py
Script 2 of 4 in the pipeline.

Job: take an uploaded file (PDF, Word, Excel, or image) and extract its
raw text. This script does NOT call any LLM — it is plain deterministic
code. (Worth remembering for Q&A: this is the one pipeline stage that
isn't "AI", it's extraction libraries doing exactly what they're told.)

Images (and scanned/photographed PDFs with no real text layer) go through
OCR via pytesseract. OCR quality on messy handwriting or low-quality
photos is NOT guaranteed — this is a known, named limitation, not
something to oversell in a demo.

Requires the system package `tesseract-ocr` to be installed for
pytesseract to work (not just the Python wrapper). This is noted in
requirements/setup instructions.
"""

import os
import logging
import pdfplumber
import docx
import openpyxl
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {"pdf", "doc", "docx", "xls", "xlsx", "png", "jpg", "jpeg", "webp"}


class ParseResult:
    """Simple result wrapper so callers always get a consistent shape back."""
    def __init__(self, success, text="", error=None, source_filename=""):
        self.success = success
        self.text = text
        self.error = error
        self.source_filename = source_filename

    def to_dict(self):
        return {
            "success": self.success,
            "text": self.text,
            "error": self.error,
            "source_filename": self.source_filename,
        }


def get_extension(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def parse_pdf(filepath):
    """
    Extract text from a PDF. If the PDF has a real text layer, pdfplumber
    gets it directly. If a page has little/no extractable text (common with
    scanned PDFs), this MVP does not auto-fallback to OCR for PDFs — only
    for standalone image files. That's a named limitation, not a secret bug:
    a scanned PDF syllabus may come back mostly empty. Flagged in README.
    """
    text_parts = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def parse_docx(filepath):
    """Extract text from a Word (.docx) file — paragraphs and table cells."""
    document = docx.Document(filepath)
    text_parts = [p.text for p in document.paragraphs if p.text.strip()]

    for table in document.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip(" |"):
                text_parts.append(row_text)

    return "\n".join(text_parts).strip()


def parse_xlsx(filepath):
    """Extract text from an Excel file — every sheet, every non-empty row."""
    workbook = openpyxl.load_workbook(filepath, data_only=True)
    text_parts = []

    for sheet in workbook.worksheets:
        text_parts.append(f"[Sheet: {sheet.title}]")
        for row in sheet.iter_rows(values_only=True):
            row_values = [str(cell) for cell in row if cell is not None]
            if row_values:
                text_parts.append(" | ".join(row_values))

    return "\n".join(text_parts).strip()


def parse_xls(filepath):
    """
    Extract text from a legacy Excel (.xls) file via xlrd — openpyxl only
    reads .xlsx. Same output shape as parse_xlsx(). Dates are converted
    from Excel serial numbers to ISO dates so exam dates stay readable
    for the syllabus-structuring step.

    Files named .xls are sometimes really .xlsx (renamed, or saved by tools
    that use the old extension). .xlsx is a zip container, so sniff the
    first bytes and hand those to parse_xlsx() instead of failing.
    """
    with open(filepath, "rb") as f:
        if f.read(2) == b"PK":
            f.seek(0)
            # Pass the open file, not the path: openpyxl rejects any *path*
            # ending in ".xls" by name, but accepts a file object.
            return parse_xlsx(f)

    import xlrd  # imported here so a missing xlrd only affects .xls uploads

    workbook = xlrd.open_workbook(filepath)
    text_parts = []

    for sheet in workbook.sheets():
        text_parts.append(f"[Sheet: {sheet.name}]")
        for row_idx in range(sheet.nrows):
            row_values = []
            for cell in sheet.row(row_idx):
                if cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
                    continue
                value = cell.value
                if cell.ctype == xlrd.XL_CELL_DATE:
                    try:
                        value = xlrd.xldate_as_datetime(value, workbook.datemode).isoformat(sep=" ")
                    except (xlrd.XLDateError, ValueError, OverflowError):
                        pass  # keep the raw number rather than dropping the cell
                elif cell.ctype == xlrd.XL_CELL_NUMBER and float(value).is_integer():
                    value = int(value)  # 3.0 -> 3
                elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                    value = bool(value)
                text = str(value).strip()
                if text:
                    row_values.append(text)
            if row_values:
                text_parts.append(" | ".join(row_values))

    return "\n".join(text_parts).strip()


def parse_image(filepath):
    """OCR an image file using pytesseract."""
    image = Image.open(filepath)
    text = pytesseract.image_to_string(image)
    return text.strip()


def parse_document(filepath, original_filename=None):
    """
    Main entry point. Looks at the file extension and routes to the right
    parser. Returns a ParseResult — always check .success before using .text.
    """
    display_name = original_filename or os.path.basename(filepath)
    ext = get_extension(display_name)

    if ext not in SUPPORTED_EXTENSIONS:
        return ParseResult(
            success=False,
            error=f"Unsupported file type: .{ext}",
            source_filename=display_name
        )

    try:
        if ext == "pdf":
            text = parse_pdf(filepath)
        elif ext == "docx":
            text = parse_docx(filepath)
        elif ext == "doc":
            # Legacy .doc (not .docx) is not supported by python-docx.
            return ParseResult(
                success=False,
                error="Old .doc format isn't supported — please upload as .docx.",
                source_filename=display_name
            )
        elif ext == "xls":
            text = parse_xls(filepath)
        elif ext == "xlsx":
            text = parse_xlsx(filepath)
        elif ext in ("png", "jpg", "jpeg", "webp"):
            text = parse_image(filepath)
        else:
            text = ""

        if not text:
            return ParseResult(
                success=False,
                error="No readable text found in this file.",
                source_filename=display_name
            )

        logger.info(f"Parsed {display_name} ({ext}) — {len(text)} chars extracted")
        return ParseResult(success=True, text=text, source_filename=display_name)

    except Exception as e:
        logger.error(f"Failed to parse {display_name}: {e}")
        return ParseResult(
            success=False,
            error=f"Couldn't read this file: {str(e)}",
            source_filename=display_name
        )
