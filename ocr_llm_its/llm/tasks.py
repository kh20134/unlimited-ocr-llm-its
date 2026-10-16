from __future__ import annotations

import json
import re

from ocr_llm_its.errors import PipelineError

TASKS = ("extract", "summarize", "qa")

_SYSTEM = {
    "extract": (
        "당신은 지능형교통체계(ITS), 요금징수, 도로 인프라 현장 문서를 정리하는 보조자입니다. "
        "OCR 마크다운만 근거로 사용하고, 문서에 없는 숫자는 만들지 마세요. "
        "반드시 JSON 객체 하나만 출력하세요. 설명 문장은 붙이지 마세요. "
        "키는 한국어로 유지하세요. 스키마는 다음과 같습니다. "
        '{"document_type":"통행영수증|점검일지|요금정산표|기타",'
        '"title":"",'
        '"fields":{"필드명":"값"},'
        '"tables":[{"name":"","columns":[],"rows":[[]]}],'
        '"notes":[]}'
    ),
    "summarize": (
        "당신은 ITS, 요금징수, 인프라 점검 문서를 한국어로 요약하는 보조자입니다. "
        "OCR 마크다운에 있는 사실만 사용하세요. "
        "5문장 안으로, 문서 종류, 장소 또는 시설, 날짜, 핵심 숫자(금액, 건수, 점검 결과)를 포함하세요. "
        "JSON이 아니라 한국어 문단으로 답하세요."
    ),
    "qa": (
        "당신은 ITS 현장 문서에 대한 질의응답 보조자입니다. "
        "아래 OCR 마크다운에 적힌 내용만으로 한국어로 답하세요. "
        "근거가 없으면 '문서에서 확인되지 않습니다.'라고 답하고 추측하지 마세요."
    ),
}


def build_messages(task: str, markdown: str, question: str | None = None) -> list[dict[str, str]]:
    if task not in TASKS:
        raise PipelineError("작업은 summarize, extract, qa 중 하나여야 합니다.")
    if task == "qa" and not (question and question.strip()):
        raise PipelineError("qa 작업에는 --question 이 필요합니다.")
    user = (
        f"작업: {task}\n"
        f"질문: {(question or '').strip()}\n\n"
        f"문서 마크다운:\n{markdown.strip()}"
    )
    return [
        {"role": "system", "content": _SYSTEM[task]},
        {"role": "user", "content": user},
    ]


def parse_json_response(text: str) -> dict:
    """Accept raw JSON or a fenced ```json block from a chat model."""
    body = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", body, re.DOTALL | re.IGNORECASE)
    if fenced:
        body = fenced.group(1).strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise PipelineError("LLM이 JSON 객체를 반환하지 않았습니다. extract 작업을 다시 실행해 보세요.")
        try:
            parsed = json.loads(body[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PipelineError("LLM 응답에서 JSON을 해석하지 못했습니다.") from exc
    if not isinstance(parsed, dict):
        raise PipelineError("extract 결과는 JSON 객체여야 합니다.")
    return parsed
