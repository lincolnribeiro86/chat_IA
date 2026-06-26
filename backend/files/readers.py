"""File content extraction for all supported types."""
import io
import logging
from typing import BinaryIO

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    "text": [".txt", ".py", ".js", ".ts", ".html", ".css", ".md", ".json", ".yaml", ".yml",
             ".xml", ".sh", ".bash", ".sql", ".csv"],
    "pdf": [".pdf"],
    "docx": [".docx"],
    "excel": [".xlsx", ".xls"],
}

ALL_SUPPORTED = [ext for exts in SUPPORTED_EXTENSIONS.values() for ext in exts]


def read_file(filename: str, data: bytes) -> tuple[str, str | None]:
    """Extract text from file bytes.
    Returns (content, error_message). On error content is '' and error is set.
    """
    name_lower = filename.lower()
    ext = "." + name_lower.rsplit(".", 1)[-1] if "." in name_lower else ""

    try:
        if ext in SUPPORTED_EXTENSIONS["text"]:
            return _read_text(filename, data)

        if ext in SUPPORTED_EXTENSIONS["pdf"]:
            return _read_pdf(filename, data)

        if ext in SUPPORTED_EXTENSIONS["docx"]:
            return _read_docx(filename, data)

        if ext in SUPPORTED_EXTENSIONS["excel"]:
            return _read_excel(filename, data)

        return "", f"Unsupported file type: {ext}"
    except Exception as e:
        logger.error(f"read_file({filename}): {e}")
        return "", str(e)


def _read_text(filename: str, data: bytes) -> tuple[str, str | None]:
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return data.decode(enc), None
        except UnicodeDecodeError:
            continue
    return "", f"Could not decode '{filename}' with any known encoding."


def _read_pdf(filename: str, data: bytes) -> tuple[str, str | None]:
    from pypdf import PdfReader, errors as pypdf_errors
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            return "", f"PDF '{filename}' has no extractable text."
        return text, None
    except pypdf_errors.PdfReadError as e:
        return "", f"Cannot read PDF '{filename}': {e}"


def _read_docx(filename: str, data: bytes) -> tuple[str, str | None]:
    from docx import Document
    try:
        doc = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs).strip()
        return text, None
    except Exception as e:
        return "", f"Cannot read DOCX '{filename}': {e}"


def _read_excel(filename: str, data: bytes) -> tuple[str, str | None]:
    import pandas as pd
    try:
        df = pd.read_excel(io.BytesIO(data))
        return df.to_string(index=False), None
    except Exception as e:
        return "", f"Cannot read Excel '{filename}': {e}"
