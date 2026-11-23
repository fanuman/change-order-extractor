"""Turn an input file (PDF or plain text) or pasted text into plain text for extraction.

This stage never raises for a bad document. An empty, scanned, encrypted or broken file comes
back with whatever text could be read plus a warning, so the pipeline can still print a valid
result that is flagged for review. Only a missing file is an error: that is a usage mistake,
not a document problem.
"""

from dataclasses import dataclass
from pathlib import Path

import pdfplumber

PDF_MAGIC = b"%PDF-"
# A PDF with fewer readable characters than this per page probably has no text layer (a scan).
MIN_CHARS_PER_PAGE = 20


@dataclass(frozen=True)
class Document:
    text: str
    source_type: str  # "pdf" or "text"
    page_count: int | None = None
    warnings: tuple[str, ...] = ()


def load(path: Path) -> Document:
    """Read a file from disk. PDFs are detected by content, not by file extension."""
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    raw = path.read_bytes()
    if PDF_MAGIC in raw[:1024]:
        return _load_pdf(path)
    return _load_text(raw)


def from_text(text: str) -> Document:
    """Pasted text, for example from stdin."""
    cleaned = clean(text)
    warnings = () if cleaned else ("document is empty",)
    return Document(text=cleaned, source_type="text", warnings=warnings)


def clean(text: str) -> str:
    """Small, safe clean-up. The same cleaned text goes to the LLM and to grounding, so quotes match."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\x00", "")
    return text.strip()


def _load_pdf(path: Path) -> Document:
    try:
        with pdfplumber.open(path) as pdf:
            pages: list[str] = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:  # pdfminer raises many error types for broken or encrypted files
        return Document(text="", source_type="pdf", warnings=(f"could not read PDF ({type(exc).__name__})",))

    text = clean("\n\n".join(pages))
    warnings: list[str] = []
    readable_chars = len("".join(text.split()))
    if readable_chars < MIN_CHARS_PER_PAGE * max(len(pages), 1):
        warnings.append("PDF has little or no text layer (possibly scanned); OCR is not supported")
    return Document(text=text, source_type="pdf", page_count=len(pages), warnings=tuple(warnings))


def _load_text(raw: bytes) -> Document:
    warnings: list[str] = []
    try:
        text = raw.decode("utf-8-sig")  # -sig also strips a byte-order mark if there is one
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
        warnings.append("file is not valid UTF-8; decoded as Windows-1252")
    text = clean(text)
    if not text:
        warnings.append("document is empty")
    return Document(text=text, source_type="text", warnings=tuple(warnings))
