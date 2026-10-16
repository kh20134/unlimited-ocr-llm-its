from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OcrResult:
    markdown: str
    backend: str
    source: str
    notice: str
