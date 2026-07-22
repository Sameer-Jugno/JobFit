"""Validate uploads and extract text from PDF or plain text."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

from pypdf import PdfReader

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
MAX_BYTES = 5 * 1024 * 1024
MAX_CHARS = 80_000
MAX_PAGES = 30
MIN_CHARS = 20


class DocumentError(ValueError):
    """User-facing validation or extraction error."""


def read_upload(filename: str, data: bytes) -> str:
    name = Path(filename or "").name.strip()
    if not name:
        raise DocumentError("Choose a file first.")

    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise DocumentError("Use a PDF, TXT, or MD file.")
    if not data:
        raise DocumentError("That file is empty.")
    if len(data) > MAX_BYTES:
        raise DocumentError("File is too large (max 5 MB).")

    if suffix == ".pdf":
        text = _pdf_text(data)
    else:
        text = _plain_text(data)

    text = _clean_extracted(text)
    if len(text) < MIN_CHARS:
        raise DocumentError(
            "Could not read enough text. Use a text PDF, or paste the content instead."
        )
    if len(text) > MAX_CHARS:
        raise DocumentError("That document is too long (max 80,000 characters).")
    return text


def _plain_text(data: bytes) -> str:
    if b"\x00" in data:
        raise DocumentError("That file looks binary, not text.")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentError("Could not read that text file as UTF-8.") from exc


def _pdf_text(data: bytes) -> str:
    if b"%PDF" not in data[:1024]:
        raise DocumentError("That file is not a valid PDF.")
    try:
        reader = PdfReader(BytesIO(data))
    except Exception as exc:
        raise DocumentError("Could not open that PDF.") from exc

    if reader.is_encrypted:
        raise DocumentError("This PDF is password-protected.")
    if not reader.pages:
        raise DocumentError("That PDF has no pages.")
    if len(reader.pages) > MAX_PAGES:
        raise DocumentError(f"That PDF has too many pages (max {MAX_PAGES}).")

    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def _clean_extracted(text: str) -> str:
    text = re.sub(r"\s+@", "@", text)
    text = re.sub(r"@\s+", "@", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
