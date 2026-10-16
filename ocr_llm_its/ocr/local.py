from __future__ import annotations

import tempfile
from pathlib import Path

from ocr_llm_its.errors import CudaRequiredError
from ocr_llm_its.ocr.clean import clean_ocr_markdown
from ocr_llm_its.ocr.types import OcrResult
from ocr_llm_its.pdfutil import pdf_to_images


class LocalUnlimitedOcr:
    """Run ``baidu/Unlimited-OCR`` with Hugging Face transformers on CUDA.

    Upstream ``infer`` moves inputs with ``.cuda()`` and the published recipe
    is an NVIDIA GPU. This class does not attempt CPU or MPS inference.

    Single images use the upstream "gundam" crop settings
    (base_size=1024, image_size=640, crop_mode=True). PDFs use ``infer_multi``
    at image_size=1024, which is the setting the upstream README allows for
    multi-page input.
    """

    name = "local"

    def __init__(
        self,
        model_name: str = "baidu/Unlimited-OCR",
        hf_token: str | None = None,
        image_mode: str = "gundam",
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.image_mode = image_mode
        self._tokenizer = None
        self._model = None

    def parse(self, path: Path | str) -> OcrResult:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {source}")
        model, tokenizer = self._load()
        output_dir = tempfile.mkdtemp(prefix="unlimited_ocr_")
        suffix = source.suffix.lower()
        if suffix == ".pdf":
            image_files = pdf_to_images(source)
            outputs, _token_count = model.infer_multi(
                tokenizer,
                prompt="<image>Multi page parsing.",
                image_files=image_files,
                output_path=output_dir,
                image_size=1024,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=1024,
                save_results=False,
            )
            raw = outputs if isinstance(outputs, str) else str(outputs)
        else:
            if self.image_mode == "base":
                base_size, image_size, crop_mode, ngram_window = 1024, 1024, False, 128
            else:
                base_size, image_size, crop_mode, ngram_window = 1024, 640, True, 128
            raw = model.infer(
                tokenizer,
                prompt="<image>document parsing.",
                image_file=str(source),
                output_path=output_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                eval_mode=True,
                save_results=False,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=ngram_window,
            )
            if not isinstance(raw, str):
                result_md = Path(output_dir) / "result.md"
                if result_md.is_file():
                    raw = result_md.read_text(encoding="utf-8")
                else:
                    raw = "" if raw is None else str(raw)
        markdown = clean_ocr_markdown(raw)
        return OcrResult(
            markdown=markdown,
            backend=self.name,
            source=str(source),
            notice=(
                f"CUDA에서 {self.model_name} 로컬 추론을 사용했습니다 "
                f"(image_mode={self.image_mode if suffix != '.pdf' else 'base-multipage'}). "
                "업스트림 모델링 코드는 trust_remote_code로 실행됩니다."
            ),
        )

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise CudaRequiredError(
                "로컬 Unlimited-OCR에는 torch와 transformers가 필요합니다. "
                "pip install -r requirements-gpu.txt 후 NVIDIA GPU에서 다시 실행하세요."
            ) from exc
        if not torch.cuda.is_available():
            raise CudaRequiredError(
                "로컬 백엔드는 NVIDIA CUDA GPU가 있어야 합니다. "
                "업스트림 infer가 입력을 .cuda()로 올리므로 CPU로는 실행하지 않습니다. "
                "GPU가 없으면 --backend fixture 를 사용하세요."
            )
        token = self.hf_token or None
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=token,
        )
        model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        model = model.eval().cuda()
        self._tokenizer = tokenizer
        self._model = model
        return model, tokenizer
