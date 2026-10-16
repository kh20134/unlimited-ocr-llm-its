from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocr_llm_its.config import Settings
from ocr_llm_its.llm.client import build_llm_client, llm_notice
from ocr_llm_its.llm.tasks import TASKS, build_messages, parse_json_response
from ocr_llm_its.ocr.select import build_ocr_backend


@dataclass
class RunResult:
    input_path: str
    task: str
    question: str | None
    ocr_backend: str
    ocr_source: str
    ocr_markdown: str
    llm_backend: str
    llm_model: str | None
    result: Any
    notice: str

    def as_dict(self) -> dict:
        return {
            "input": self.input_path,
            "task": self.task,
            "question": self.question,
            "ocr": {
                "backend": self.ocr_backend,
                "source": self.ocr_source,
                "markdown": self.ocr_markdown,
            },
            "llm": {
                "backend": self.llm_backend,
                "model": self.llm_model,
            },
            "result": self.result,
            "notice": self.notice,
        }


def run_document(
    path: Path | str,
    task: str,
    question: str | None = None,
    *,
    ocr_backend: str | None = None,
    llm_backend: str | None = None,
    settings: Settings | None = None,
    llm_client=None,
    ollama_up: bool | None = None,
) -> RunResult:
    """OCR a document, then run extract, summarize, or qa."""
    if task not in TASKS:
        from ocr_llm_its.errors import PipelineError

        raise PipelineError("작업은 summarize, extract, qa 중 하나여야 합니다.")
    settings = settings or Settings.from_env()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {source}")

    ocr = build_ocr_backend(ocr_backend or settings.ocr_backend, settings)
    ocr_result = ocr.parse(source)
    messages = build_messages(task, ocr_result.markdown, question)
    client = llm_client if llm_client is not None else build_llm_client(
        llm_backend or settings.llm_backend,
        settings,
        ollama_up=ollama_up,
    )
    raw = client.complete(messages)
    if task == "extract":
        parsed: Any = parse_json_response(raw)
    else:
        parsed = (raw or "").strip()
    notice = ocr_result.notice
    extra = llm_notice(client)
    if extra:
        notice = notice + " " + extra
    return RunResult(
        input_path=str(source),
        task=task,
        question=question.strip() if question and question.strip() else None,
        ocr_backend=ocr_result.backend,
        ocr_source=ocr_result.source,
        ocr_markdown=ocr_result.markdown,
        llm_backend=getattr(client, "name", "custom"),
        llm_model=getattr(client, "model", None),
        result=parsed,
        notice=notice.strip(),
    )
