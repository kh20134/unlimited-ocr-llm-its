from __future__ import annotations

import tempfile
from pathlib import Path


def pdf_to_images(pdf_path: Path | str, dpi: int = 144) -> list[str]:
    """Rasterize a PDF the same way the Unlimited-OCR README does, via PyMuPDF.

    The local GPU backend is the only caller. Importing PyMuPDF stays inside
    this function so fixture-only runs do not need it.
    """
    import fitz

    doc = fitz.open(pdf_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix="pdf_ocr_"))
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    paths: list[str] = []
    try:
        for index, page in enumerate(doc):
            out = tmp_dir / f"page_{index + 1:04d}.png"
            page.get_pixmap(matrix=matrix).save(out)
            paths.append(str(out))
    finally:
        doc.close()
    if not paths:
        raise ValueError(f"PDF에 페이지가 없습니다: {pdf_path}")
    return paths
