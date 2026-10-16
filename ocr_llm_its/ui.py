from __future__ import annotations

import json
from pathlib import Path

from ocr_llm_its.config import Settings, repo_root
from ocr_llm_its.errors import PipelineError
from ocr_llm_its.pipeline import run_document


def sample_files() -> list[str]:
    root = repo_root() / "samples"
    names = ["toll_receipt.png", "inspection_log.png", "settlement_table.pdf"]
    return [str(root / name) for name in names if (root / name).is_file()]


def _as_path(uploaded) -> str | None:
    if uploaded is None:
        return None
    if isinstance(uploaded, str):
        return uploaded
    name = getattr(uploaded, "name", None)
    return str(name) if name else None


def run_from_ui(sample: str, uploaded, task: str, question: str, backend: str, llm: str) -> tuple[str, str, str]:
    path = _as_path(uploaded) or sample
    if not path:
        return "", "", "샘플을 고르거나 파일을 올려 주세요."
    try:
        result = run_document(
            path,
            task,
            question,
            ocr_backend=backend,
            llm_backend=llm,
            settings=Settings.from_env(),
        )
    except (PipelineError, FileNotFoundError, OSError) as exc:
        return "", "", str(exc)
    pretty = result.result
    if not isinstance(pretty, str):
        pretty = json.dumps(pretty, ensure_ascii=False, indent=2)
    return result.ocr_markdown, pretty, result.notice


def launch(server_port: int = 7860, share: bool = False) -> None:
    import os

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    samples = sample_files()
    diagram = repo_root() / "docs" / "images" / "architecture.png"
    with gr.Blocks(title="ITS 현장 문서 도우미") as demo:
        gr.Markdown(
            "# ITS 현장 문서 도우미\n"
            "이미지 또는 PDF를 Unlimited-OCR(GPU) 또는 샘플 픽스처로 읽은 뒤, "
            "구조화 JSON, 한국어 요약, 문서 질의응답을 만듭니다. "
            "GPU와 API 키가 없어도 샘플 세 개는 끝까지 실행됩니다."
        )
        if diagram.is_file():
            gr.Image(value=str(diagram), label="파이프라인 구조", interactive=False)
        with gr.Row():
            sample = gr.Dropdown(choices=samples, value=samples[0] if samples else None, label="샘플 문서")
            task = gr.Radio(
                choices=["summarize", "extract", "qa"],
                value="summarize",
                label="작업",
            )
        question = gr.Textbox(label="질문 (qa일 때)", placeholder="예: 통행료는 얼마인가?")
        upload = gr.File(label="또는 내 파일 올리기 (GPU가 없으면 픽스처가 있는 샘플만 동작)")
        with gr.Row():
            backend = gr.Dropdown(choices=["auto", "fixture", "local"], value="auto", label="OCR 백엔드")
            llm = gr.Dropdown(
                choices=["auto", "offline", "openai", "ollama"],
                value="auto",
                label="LLM 백엔드",
            )
        button = gr.Button("실행")
        ocr_out = gr.Textbox(label="OCR 마크다운", lines=12)
        result_out = gr.Textbox(label="결과", lines=12)
        notice = gr.Textbox(label="안내", lines=3)
        button.click(
            run_from_ui,
            inputs=[sample, upload, task, question, backend, llm],
            outputs=[ocr_out, result_out, notice],
        )
    demo.launch(server_name="127.0.0.1", server_port=server_port, share=share)
