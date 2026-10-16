from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ocr_llm_its import __version__
from ocr_llm_its.config import Settings
from ocr_llm_its.errors import PipelineError
from ocr_llm_its.pipeline import run_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr_llm_its",
        description="ITS 현장 문서용 Unlimited-OCR + LLM 파이프라인",
    )
    parser.add_argument("--version", action="version", version=f"ocr_llm_its {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="이미지 또는 PDF를 OCR한 뒤 요약, 추출, 질의응답을 실행")
    run.add_argument("--input", required=True, help="이미지 또는 PDF 경로")
    run.add_argument("--task", required=True, choices=["summarize", "extract", "qa"])
    run.add_argument("--question", default=None, help="qa 작업의 질문")
    run.add_argument("--backend", default=None, choices=["auto", "local", "fixture"], help="OCR 백엔드")
    run.add_argument("--llm", default=None, choices=["auto", "openai", "ollama", "offline"])
    run.add_argument("--output", default=None, help="결과 JSON을 이 파일에도 저장")

    ui = sub.add_parser("ui", help="로컬 Gradio 화면을 엽니다")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--share", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ui":
            from ocr_llm_its.ui import launch

            launch(server_port=args.port, share=args.share)
            return 0
        settings = Settings.from_env()
        result = run_document(
            args.input,
            args.task,
            args.question,
            ocr_backend=args.backend,
            llm_backend=args.llm,
            settings=settings,
        )
    except (PipelineError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    payload = result.as_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)
    if result.notice:
        print(result.notice, file=sys.stderr)
    return 0
