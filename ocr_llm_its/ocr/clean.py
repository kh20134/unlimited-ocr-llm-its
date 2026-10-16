from __future__ import annotations

import re

# Upstream post-process from the Unlimited-OCR README (MIT, Copyright (c) 2026 Baidu).
# Markers look like: <|det|>text [x1, y1, x2, y2]<|/det|>본문
DET_RE = re.compile(
    r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)",
    re.DOTALL,
)

_STOP_STRINGS = (
    "<｜end▁of▁sentence｜>",
    "<|end▁of▁sentence|>",
)


def remove_det(raw: str) -> str:
    """Drop detection markers and keep readable markdown blocks."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    saw_marker = False
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        match = DET_RE.match(line)
        if match:
            saw_marker = True
            category, content = match.group(1).strip(), match.group(2).strip()
            if category == "image":
                continue
            if current is not None:
                blocks.append(current)
            current = [content] if content else []
            continue
        if current is None:
            current = []
        current.append(line)
    if not saw_marker:
        return raw
    if current is not None:
        blocks.append(current)
    return "\n\n".join("\n".join(block) for block in blocks).strip()


def clean_ocr_markdown(raw: str) -> str:
    """Normalize Unlimited-OCR text before it is sent to the LLM."""
    text = raw or ""
    for stop in _STOP_STRINGS:
        if text.endswith(stop):
            text = text[: -len(stop)]
    text = remove_det(text)
    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:").strip()
    return text
