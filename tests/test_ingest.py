from pathlib import Path

import pytest

from co_extract.ingest import Document, from_text, load


def make_pdf(lines: list[str]) -> bytes:
    """A one-page PDF with a real text layer, built by hand so the tests need no PDF library.

    With no lines the page has no text at all, which is what a scanned PDF looks like to us.
    """

    def escape(line: str) -> str:
        return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    shown = " ".join(f"({escape(line)}) Tj T*" for line in lines)
    content = f"BT /F1 12 Tf 72 720 Td 14 TL {shown} ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_at = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return pdf


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --- plain text ---------------------------------------------------------------------------


def test_text_file_is_cleaned(tmp_path: Path) -> None:
    path = write(tmp_path, "co.txt", "CHANGE ORDER No. 7\r\nAmount:\u00a0$1,250.00\x00\r\n".encode())
    doc = load(path)
    assert doc == Document(text="CHANGE ORDER No. 7\nAmount: $1,250.00", source_type="text")


def test_utf8_byte_order_mark_is_stripped(tmp_path: Path) -> None:
    path = write(tmp_path, "co.txt", "\ufeffChange Order 3".encode())
    assert load(path).text == "Change Order 3"


def test_windows_1252_file_is_decoded_with_a_warning(tmp_path: Path) -> None:
    path = write(tmp_path, "co.txt", "Owner\u2019s request".encode("cp1252"))
    doc = load(path)
    assert doc.text == "Owner\u2019s request"
    assert doc.warnings == ("file is not valid UTF-8; decoded as Windows-1252",)


def test_empty_text_file_warns(tmp_path: Path) -> None:
    doc = load(write(tmp_path, "empty.txt", b"  \n\r\n "))
    assert doc.text == ""
    assert doc.warnings == ("document is empty",)


def test_pasted_text() -> None:
    assert from_text("  Change Order 12\r\n") == Document(text="Change Order 12", source_type="text")


def test_empty_pasted_text_warns() -> None:
    assert from_text("   ").warnings == ("document is empty",)


# --- PDF ----------------------------------------------------------------------------------


def test_pdf_text_is_extracted(tmp_path: Path) -> None:
    doc = load(write(tmp_path, "co.pdf", make_pdf(["CHANGE ORDER No. 7", "Amount: (1,250.00)"])))
    assert doc.source_type == "pdf"
    assert doc.page_count == 1
    assert "CHANGE ORDER No. 7" in doc.text
    assert "Amount: (1,250.00)" in doc.text
    assert doc.warnings == ()


def test_pdf_is_detected_by_content_not_extension(tmp_path: Path) -> None:
    doc = load(write(tmp_path, "looks-like-text.txt", make_pdf(["CHANGE ORDER No. 7 for the new school wing"])))
    assert doc.source_type == "pdf"


def test_pdf_without_text_layer_warns_possibly_scanned(tmp_path: Path) -> None:
    doc = load(write(tmp_path, "scan.pdf", make_pdf([])))
    assert doc.source_type == "pdf"
    assert doc.text == ""
    assert len(doc.warnings) == 1
    assert "possibly scanned" in doc.warnings[0]


def test_broken_pdf_does_not_crash(tmp_path: Path) -> None:
    doc = load(write(tmp_path, "broken.pdf", b"%PDF-1.4\nthis is not really a pdf"))
    assert doc.text == ""
    assert len(doc.warnings) == 1
    assert doc.warnings[0].startswith("could not read PDF")


# --- usage errors -------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.pdf")
