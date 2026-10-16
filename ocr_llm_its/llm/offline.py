from __future__ import annotations

import json
import re

_FIELD_RE = re.compile(r"^(?:[-*]\s*)?([^:：|#]{1,40})[:：]\s+(.+)$")
_DOC_SPLIT = "문서 마크다운:\n"


def parse_markdown_document(markdown: str) -> dict:
    """Turn OCR-style markdown into title, ordered fields, tables, and notes."""
    title = ""
    section = ""
    fields: dict[str, str] = {}
    tables: list[dict] = []
    notes: list[str] = []
    blocks: list[dict] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            index += 1
            continue
        if stripped.startswith("## "):
            section = stripped[3:].strip()
            index += 1
            continue
        if stripped.startswith("|"):
            block: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index].strip())
                index += 1
            table = _parse_table(block, section or "표")
            if table["columns"]:
                tables.append(table)
                blocks.append({"type": "table", **table})
            section = ""
            continue
        match = _FIELD_RE.match(stripped)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            fields[key] = value
            blocks.append({"type": "field", "key": key, "value": value})
            index += 1
            continue
        notes.append(stripped)
        blocks.append({"type": "note", "text": stripped})
        index += 1
    return {"title": title, "fields": fields, "tables": tables, "notes": notes, "blocks": blocks}


def extract_document(markdown: str) -> dict:
    parsed = parse_markdown_document(markdown)
    return {
        "document_type": guess_document_type(parsed["title"], markdown),
        "title": parsed["title"],
        "fields": parsed["fields"],
        "tables": parsed["tables"],
        "notes": parsed["notes"],
    }


def guess_document_type(title: str, markdown: str) -> str:
    text = f"{title}\n{markdown}"
    if "점검" in text:
        return "점검일지"
    if "정산" in text:
        return "요금정산표"
    if "영수증" in text or "통행료" in text:
        return "통행영수증"
    return "기타"


def summarize_document(markdown: str) -> str:
    parsed = parse_markdown_document(markdown)
    title = parsed["title"] or "현장 문서"
    kind = guess_document_type(title, markdown)
    chosen: list[str] = []
    seen: set[str] = set()
    ordered = list(parsed["fields"].items())
    highlighted = [
        (key, value)
        for key, value in ordered
        if any(token in key for token in ("통행료", "금액", "합계", "결제", "결과")) or value.endswith("원")
    ]
    for key, value in ordered[:6] + highlighted:
        if key in seen:
            continue
        seen.add(key)
        chosen.append(f"{key} {value}")
    sentences = [f"문서 제목은 {title}이고 종류는 {kind}입니다."]
    if chosen:
        sentences.append("주요 항목은 " + ", ".join(chosen) + "입니다.")
    for table in parsed["tables"]:
        sentences.append(
            f"{table['name']} 표에는 {len(table['rows'])}개 행이 있고 열은 {', '.join(table['columns'])}입니다."
        )
    if parsed["notes"]:
        sentences.append(parsed["notes"][0])
    return " ".join(sentences)


def answer_question(markdown: str, question: str) -> str:
    parsed = parse_markdown_document(markdown)
    asked = (question or "").strip()
    if not asked:
        return "문서에서 확인되지 않습니다."
    ranked = sorted(parsed["fields"], key=len, reverse=True)
    for key in ranked:
        if key and key in asked:
            return f"{key}의 값은 {parsed['fields'][key]}입니다."
    for table in parsed["tables"]:
        for row in table["rows"]:
            if not row:
                continue
            if row[0] and row[0] in asked:
                pairs = [f"{col} {val}" for col, val in zip(table["columns"], row)]
                return f"{table['name']}의 {row[0]} 행은 " + ", ".join(pairs) + "입니다."
        for column in table["columns"]:
            if column and column in asked:
                preview = "; ".join(
                    ", ".join(f"{col} {val}" for col, val in zip(table["columns"], row))
                    for row in table["rows"][:5]
                )
                return f"{table['name']}의 {column} 열을 포함한 행은 다음과 같습니다. {preview}"
    for note in parsed["notes"]:
        if any(token and token in note for token in _tokens(asked)):
            return note
    for key, value in parsed["fields"].items():
        if value and value in asked:
            return f"{key}의 값은 {value}입니다."
    return "문서에서 확인되지 않습니다."


class OfflineChatClient:
    """Deterministic stand-in used when no OpenAI-compatible server or Ollama is configured.

    It is not a neural LLM. It reads the same chat messages the real client would
    send and fills the three tasks from the markdown so a beginner can finish
    the demo without a GPU or an API key.
    """

    name = "offline"
    model = None

    def complete(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        task = _task_from_messages(system, user)
        question = _field_from_user(user, "질문")
        markdown = user.split(_DOC_SPLIT, 1)[1] if _DOC_SPLIT in user else user
        if task == "extract":
            return json.dumps(extract_document(markdown), ensure_ascii=False)
        if task == "qa":
            return answer_question(markdown, question)
        return summarize_document(markdown)


def _parse_table(lines: list[str], name: str) -> dict:
    rows = [_split_row(line) for line in lines]
    rows = [row for row in rows if row and not _is_separator(row)]
    if not rows:
        return {"name": name, "columns": [], "rows": []}
    columns = rows[0]
    body = rows[1:]
    return {"name": name, "columns": columns, "rows": body}


def _split_row(line: str) -> list[str]:
    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return [cell.strip() for cell in trimmed.split("|")]


def _is_separator(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells if cell)


def _task_from_messages(system: str, user: str) -> str:
    match = re.search(r"작업:\s*(extract|summarize|qa)", user)
    if match:
        return match.group(1)
    if "JSON" in system:
        return "extract"
    if "질의" in system:
        return "qa"
    return "summarize"


def _field_from_user(user: str, label: str) -> str:
    match = re.search(rf"{label}:\s*(.*)", user)
    return match.group(1).strip() if match else ""


def _tokens(question: str) -> list[str]:
    rough = re.sub(r"[은는이가을를에의\?\!\.,]", " ", question)
    return [part for part in rough.split() if len(part) >= 2]

