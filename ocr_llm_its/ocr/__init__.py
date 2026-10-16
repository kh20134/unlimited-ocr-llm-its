"""OCR backends: local Unlimited-OCR, or precomputed fixtures when there is no GPU."""

from ocr_llm_its.ocr.clean import clean_ocr_markdown
from ocr_llm_its.ocr.select import build_ocr_backend, cuda_available

__all__ = ["build_ocr_backend", "clean_ocr_markdown", "cuda_available"]
