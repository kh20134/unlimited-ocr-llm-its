from __future__ import annotations

from pathlib import Path

from ocr_llm_its.errors import FixtureNotFoundError
from ocr_llm_its.ocr.types import OcrResult


class FixtureOcr:
    """Read precomputed markdown so the LLM stage can run with no GPU and no download."""

    name = "fixture"

    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = Path(fixtures_dir)

    def parse(self, path: Path | str) -> OcrResult:
        source = Path(path)
        fixture = self.fixtures_dir / f"{source.stem}.md"
        if not fixture.is_file():
            raise FixtureNotFoundError(
                "이 파일에 대응하는 픽스처 마크다운이 없습니다: "
                f"{fixture}. GPU에서 --backend local 로 실행하거나 "
                "samples/fixtures/<파일명>.md 를 추가하세요."
            )
        markdown = fixture.read_text(encoding="utf-8").strip()
        if not markdown:
            raise FixtureNotFoundError(f"픽스처가 비어 있습니다: {fixture}")
        return OcrResult(
            markdown=markdown,
            backend=self.name,
            source=str(fixture),
            notice=(
                "CUDA GPU가 없거나 픽스처 백엔드를 지정해, 미리 적어 둔 OCR 마크다운을 사용합니다. "
                "baidu/Unlimited-OCR 가중치는 받지 않습니다. "
                "이 마크다운은 샘플 문서와 같은 내용의 자리표시자이며, 모델이 이 환경에서 생성한 출력이 아닙니다."
            ),
        )
