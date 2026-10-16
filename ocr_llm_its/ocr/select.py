from __future__ import annotations

from ocr_llm_its.config import Settings
from ocr_llm_its.errors import CudaRequiredError, PipelineError
from ocr_llm_its.ocr.fixture import FixtureOcr
from ocr_llm_its.ocr.local import LocalUnlimitedOcr


def cuda_available() -> bool:
    """True only when PyTorch imports and sees a CUDA device.

    A missing torch install counts as "no GPU" so the fixture path still runs
    on a fresh laptop.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build_ocr_backend(choice: str, settings: Settings):
    """Pick a backend.

    ``auto`` uses local Unlimited-OCR when CUDA is available and otherwise
    falls back to fixtures. ``local`` never falls back silently: it errors
    when CUDA is missing. ``fixture`` always reads precomputed markdown.
    """
    selected = (choice or "auto").strip().lower()
    if selected not in {"auto", "local", "fixture"}:
        raise PipelineError("OCR 백엔드는 auto, local, fixture 중 하나여야 합니다.")
    if selected == "fixture":
        return FixtureOcr(settings.fixtures_dir)
    if selected == "local":
        if not cuda_available():
            raise CudaRequiredError(
                "OCR 백엔드 local을 요청했지만 CUDA를 사용할 수 없습니다. "
                "가중치(약 3.3B 파라미터, BF16 파일 약 6.8GB)는 NVIDIA GPU에서만 실행합니다. "
                "데모는 --backend fixture 로 실행하세요."
            )
        return LocalUnlimitedOcr(model_name=settings.model_name, hf_token=settings.hf_token)
    if cuda_available():
        return LocalUnlimitedOcr(model_name=settings.model_name, hf_token=settings.hf_token)
    return FixtureOcr(settings.fixtures_dir)
