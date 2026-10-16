"""Render synthetic ITS sample pages and the Korean architecture diagrams.

Requires Noto Sans CJK (Debian/Ubuntu package ``fonts-noto-cjk``) and Pillow.
The sample pages are drawn from ``samples/fixtures/*.md`` so the picture and
the precomputed OCR text stay aligned. Run from the repository root:

    python scripts/render_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr_llm_its.llm.offline import parse_markdown_document  # noqa: E402

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
# TTC index 1 is Noto Sans CJK KR (0=JP, 1=KR, 2=SC, 3=TC, 4=HK).
FONT_INDEX = 1

PAGE_W, PAGE_H = 1240, 1754


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REG
    if not Path(path).is_file():
        raise SystemExit(
            "Noto Sans CJK 폰트가 없습니다. Ubuntu에서는 fonts-noto-cjk 패키지를 설치하세요."
        )
    return ImageFont.truetype(path, size, index=FONT_INDEX)


def wrap(text: str, face: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for ch in paragraph:
            trial = current + ch
            width = face.getbbox(trial)[2]
            if width > max_width and current:
                lines.append(current)
                current = ch
            else:
                current = trial
        lines.append(current)
    return lines or [""]


def text_size(face: ImageFont.FreeTypeFont, text: str) -> tuple[int, int]:
    box = face.getbbox(text or " ")
    return box[2] - box[0], box[3] - box[1]


def draw_page(markdown: str, theme: tuple[int, int, int], kicker: str) -> Image.Image:
    parsed = parse_markdown_document(markdown)
    image = Image.new("RGB", (PAGE_W, PAGE_H), "#F7F5F1")
    draw = ImageDraw.Draw(image)
    title_font = font(54, bold=True)
    kicker_font = font(26, bold=True)
    label_font = font(28, bold=True)
    value_font = font(28)
    small_font = font(24)
    head_font = font(26, bold=True)
    cell_font = font(24)

    draw.rectangle([0, 0, PAGE_W, 210], fill=theme)
    draw.text((64, 42), kicker, font=kicker_font, fill="#F3D7A1")
    draw.text((64, 96), parsed["title"] or "현장 문서", font=title_font, fill="white")
    badge = "견본"
    badge_w, _ = text_size(kicker_font, badge)
    draw.rounded_rectangle([PAGE_W - 64 - badge_w - 36, 48, PAGE_W - 64, 100], radius=8, fill="#F3D7A1")
    draw.text((PAGE_W - 64 - badge_w - 18, 56), badge, font=kicker_font, fill=theme)

    y = 250
    left = 64
    right = PAGE_W - 64
    for block in parsed["blocks"]:
        if y > PAGE_H - 140:
            break
        if block["type"] == "field":
            draw.rounded_rectangle([left, y, right, y + 64], radius=8, fill="white", outline="#E2DDD4")
            draw.text((left + 24, y + 16), block["key"], font=label_font, fill=theme)
            value_lines = wrap(block["value"], value_font, 680)
            draw.text((left + 340, y + 16), value_lines[0], font=value_font, fill="#1C1A17")
            y += 76
            continue
        if block["type"] == "note":
            for line in wrap(block["text"], small_font, right - left):
                draw.text((left, y), line, font=small_font, fill="#3A342C")
                y += 36
            y += 12
            continue
        table = block
        y += 8
        draw.text((left, y), table["name"], font=head_font, fill=theme)
        y += 46
        columns = table["columns"]
        col_w = (right - left) // max(len(columns), 1)
        header_h = 52
        draw.rectangle([left, y, right, y + header_h], fill=theme)
        for index, column in enumerate(columns):
            draw.text((left + index * col_w + 16, y + 12), column, font=head_font, fill="white")
        y += header_h
        for row_index, row in enumerate(table["rows"]):
            row_h = 50
            fill = "#FFFFFF" if row_index % 2 == 0 else "#F3EFE8"
            draw.rectangle([left, y, right, y + row_h], fill=fill, outline="#E2DDD4")
            for index, cell in enumerate(row):
                draw.text((left + index * col_w + 16, y + 12), cell, font=cell_font, fill="#1C1A17")
            y += row_h
        y += 24
    draw.rectangle([0, PAGE_H - 72, PAGE_W, PAGE_H], fill=theme)
    draw.text(
        (64, PAGE_H - 48),
        "합성 견본 · 실제 요금 징수나 시설 기록이 아닙니다",
        font=small_font,
        fill="white",
    )
    return image


def save_pdf(image: Image.Image, path: Path) -> None:
    import pymupdf

    png_path = path.with_suffix(".png")
    image.save(png_path, format="PNG")
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, filename=str(png_path))
    doc.save(path)
    doc.close()
    png_path.unlink()


def rounded_card(draw: ImageDraw.ImageDraw, box, fill, outline="#D5DDE6", radius=18):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def draw_center(draw, box, text, face, fill, spacing=6) -> None:
    x0, y0, x1, y1 = box
    lines = wrap(text, face, (x1 - x0) - 28)
    heights = [text_size(face, line)[1] for line in lines]
    total = sum(heights) + spacing * (len(lines) - 1)
    y = y0 + ((y1 - y0) - total) / 2
    for line, height in zip(lines, heights):
        width, _ = text_size(face, line)
        draw.text((x0 + ((x1 - x0) - width) / 2, y), line, font=face, fill=fill)
        y += height + spacing


def arrow(draw, start, end, fill="#5C6B7A"):
    draw.line([start, end], fill=fill, width=4)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        head = [
            (ex, ey),
            (ex - 14 * direction, ey - 8),
            (ex - 14 * direction, ey + 8),
        ]
    else:
        direction = 1 if ey > sy else -1
        head = [
            (ex, ey),
            (ex - 8, ey - 14 * direction),
            (ex + 8, ey - 14 * direction),
        ]
    draw.polygon(head, fill=fill)


def render_architecture(path: Path) -> None:
    width, height = 1800, 1040
    image = Image.new("RGB", (width, height), "#F4F7FB")
    draw = ImageDraw.Draw(image)
    title_font = font(42, bold=True)
    sub_font = font(22)
    card_title = font(28, bold=True)
    body = font(20)
    small = font(18)
    tiny = font(16)

    draw.text((64, 36), "ITS 현장 문서 파이프라인", font=title_font, fill="#12324F")
    draw.text(
        (64, 96),
        "이미지·PDF를 Unlimited-OCR로 읽고, LLM이 표와 요약을 만듭니다. GPU가 없으면 샘플 픽스처로 이어집니다.",
        font=sub_font,
        fill="#3E5166",
    )

    cards = [
        (48, 180, 330, 860),
        (390, 180, 730, 860),
        (790, 180, 1040, 860),
        (1100, 180, 1400, 860),
        (1460, 180, 1752, 860),
    ]
    for box in cards:
        rounded_card(draw, box, "white")

    # 1. source document
    draw.text((72, 204), "1. 현장 문서", font=card_title, fill="#12324F")
    draw.text((72, 252), "이미지 또는 PDF", font=body, fill="#3E5166")
    samples = [
        (72, 320, 306, 430, "요금 영수증", "통행료, 차종, 하이패스"),
        (72, 450, 306, 560, "점검일지", "VMS, 통신, 조치"),
        (72, 580, 306, 690, "요금정산표", "차종별 건수와 금액"),
    ]
    for x0, y0, x1, y1, name, desc in samples:
        rounded_card(draw, (x0, y0, x1, y1), "#E8F1FA", outline="#C5D7EA", radius=12)
        draw.text((x0 + 16, y0 + 22), name, font=card_title, fill="#12324F")
        draw.text((x0 + 16, y0 + 64), desc, font=small, fill="#3E5166")
    draw.text((72, 740), "합성 견본만 포함합니다.", font=small, fill="#5C6B7A")
    draw.text((72, 772), "재학습은 하지 않습니다.", font=small, fill="#5C6B7A")

    # 2. OCR branch
    draw.text((414, 204), "2. OCR 분기", font=card_title, fill="#12324F")
    draw.text((414, 252), "GPU 유무로 백엔드를 고릅니다.", font=body, fill="#3E5166")
    rounded_card(draw, (414, 320, 706, 520), "#E5F6F1", outline="#B7E0D4", radius=14)
    draw.text((434, 344), "GPU 있음", font=card_title, fill="#0F6B56")
    draw.text((434, 396), "Unlimited-OCR", font=body, fill="#12324F")
    draw.text((434, 432), "로컬 transformers", font=small, fill="#3E5166")
    draw.text((434, 464), "CUDA, 약 6.8GB 가중치", font=small, fill="#3E5166")
    rounded_card(draw, (414, 550, 706, 760), "#FFF6E8", outline="#F0D7A8", radius=14)
    draw.text((434, 574), "GPU 없음", font=card_title, fill="#8A5A10")
    draw.text((434, 626), "샘플 픽스처", font=body, fill="#12324F")
    draw.text((434, 662), "미리 적은 마크다운", font=small, fill="#3E5166")
    draw.text((434, 694), "큰 모델을 받지 않음", font=small, fill="#3E5166")

    # 3. markdown
    draw.text((814, 204), "3. 마크다운", font=card_title, fill="#12324F")
    for index, line in enumerate(["제목과 문단", "키와 값", "표", "검출 태그 제거"]):
        y = 300 + index * 90
        rounded_card(draw, (814, y, 1016, y + 70), "#F7F9FC", outline="#D5DDE6", radius=12)
        draw_center(draw, (814, y, 1016, y + 70), line, body, "#12324F")

    # 4. LLM
    draw.text((1124, 204), "4. LLM 후처리", font=card_title, fill="#12324F")
    draw.text((1124, 258), "같은 프롬프트 인터페이스", font=small, fill="#3E5166")
    modes = [
        (1124, 330, 1376, 450, "OpenAI 호환 API", "OPENAI_API_BASE / KEY"),
        (1124, 470, 1376, 590, "Ollama", "OLLAMA_HOST"),
        (1124, 610, 1376, 760, "오프라인 데모", "키 없을 때의 규칙 정리"),
    ]
    for x0, y0, x1, y1, name, desc in modes:
        rounded_card(draw, (x0, y0, x1, y1), "#F3EEFF", outline="#D9CEF5", radius=12)
        draw.text((x0 + 16, y0 + 24), name, font=card_title, fill="#3C2A6D")
        draw.text((x0 + 16, y0 + 70), desc, font=small, fill="#3E5166")

    # 5. results
    draw.text((1484, 204), "5. 결과", font=card_title, fill="#12324F")
    results = [
        (1484, 300, 1728, 450, "구조화 JSON", "표와 필드"),
        (1484, 480, 1728, 620, "한국어 요약", "핵심 숫자 포함"),
        (1484, 650, 1728, 800, "문서 질의응답", "본문에 있는 내용만"),
    ]
    for x0, y0, x1, y1, name, desc in results:
        rounded_card(draw, (x0, y0, x1, y1), "#E7F2FB", outline="#C5D7EA", radius=12)
        draw.text((x0 + 16, y0 + 28), name, font=card_title, fill="#12324F")
        draw.text((x0 + 16, y0 + 76), desc, font=small, fill="#3E5166")

    arrow(draw, (330, 520), (390, 520))
    arrow(draw, (730, 520), (790, 520))
    arrow(draw, (1040, 520), (1100, 520))
    arrow(draw, (1400, 520), (1460, 520))

    draw.text(
        (64, 960),
        "Baidu Unlimited-OCR (MIT, Copyright (c) 2026 Baidu) 를 호출합니다. 가중치와 모델링 코드는 업스트림 것입니다.",
        font=tiny,
        fill="#5C6B7A",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def render_sequence(path: Path) -> None:
    width, height = 1800, 980
    image = Image.new("RGB", (width, height), "#F4F7FB")
    draw = ImageDraw.Draw(image)
    title_font = font(40, bold=True)
    sub_font = font(22)
    actor_font = font(24, bold=True)
    label_font = font(20)

    draw.text((64, 36), "요청 한 번의 흐름", font=title_font, fill="#12324F")
    draw.text(
        (64, 96),
        "CLI와 웹 화면은 같은 파이프라인을 호출합니다. OCR이 마크다운을 주면 LLM이 작업을 수행합니다.",
        font=sub_font,
        fill="#3E5166",
    )

    actors = [
        (180, "사용자"),
        (620, "CLI · 웹"),
        (1060, "OCR"),
        (1500, "LLM"),
    ]
    for x, name in actors:
        rounded_card(draw, (x - 110, 180, x + 110, 250), "#12324F", outline="#12324F", radius=14)
        draw_center(draw, (x - 110, 180, x + 110, 250), name, actor_font, "white")
        draw.line([(x, 250), (x, 900)], fill="#C5D0DC", width=3)

    steps = [
        (180, 620, 340, "문서와 작업 전달"),
        (620, 1060, 450, "백엔드 선택 후 파싱"),
        (1060, 620, 560, "마크다운 반환"),
        (620, 1500, 690, "본문과 프롬프트"),
        (1500, 620, 800, "JSON · 요약 · 답변"),
    ]
    for start, end, y, label in steps:
        arrow(draw, (start, y), (end, y), fill="#1F6F8B")
        face_w, face_h = text_size(label_font, label)
        mid = (start + end) / 2
        box = (mid - face_w / 2 - 12, y - face_h - 22, mid + face_w / 2 + 12, y - 8)
        rounded_card(draw, box, "white", outline="#F4F7FB", radius=8)
        draw.text((box[0] + 12, box[1] + 4), label, font=label_font, fill="#12324F")

    note = "GPU면 Unlimited-OCR, 없으면 샘플 픽스처"
    note_w, _note_h = text_size(label_font, note)
    note_box = (1060 - note_w / 2 - 14, 908, 1060 + note_w / 2 + 14, 968)
    rounded_card(draw, note_box, "#FFF6E8", outline="#F0D7A8", radius=10)
    draw.text((note_box[0] + 14, note_box[1] + 10), note, font=label_font, fill="#8A5A10")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def main() -> None:
    fixtures = ROOT / "samples" / "fixtures"
    samples = ROOT / "samples"
    pages = {
        "toll_receipt": ((18, 58, 102), "도로샘플 요금소"),
        "inspection_log": ((18, 92, 74), "도로샘플 시설관리"),
        "settlement_table": ((122, 58, 36), "도로샘플 요금정산"),
    }
    for stem, (theme, kicker) in pages.items():
        markdown = (fixtures / f"{stem}.md").read_text(encoding="utf-8")
        image = draw_page(markdown, theme, kicker)
        if stem == "settlement_table":
            save_pdf(image, samples / "settlement_table.pdf")
        else:
            image.save(samples / f"{stem}.png", format="PNG")
    docs = ROOT / "docs" / "images"
    render_architecture(docs / "architecture.png")
    render_sequence(docs / "sequence.png")
    print("wrote samples and docs/images")


if __name__ == "__main__":
    main()
