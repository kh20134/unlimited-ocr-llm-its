import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr_llm_its.config import Settings
from ocr_llm_its.errors import LlmConfigError, PipelineError
from ocr_llm_its.llm.client import OpenAICompatibleClient, build_llm_client
from ocr_llm_its.llm.offline import OfflineChatClient, extract_document
from ocr_llm_its.llm.tasks import build_messages, parse_json_response


def settings(**overrides) -> Settings:
    base = dict(
        hf_token=None,
        openai_api_base="https://api.openai.com/v1",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="qwen2.5:7b",
        ocr_backend="fixture",
        llm_backend="auto",
        model_name="baidu/Unlimited-OCR",
        fixtures_dir=ROOT / "samples" / "fixtures",
    )
    base.update(overrides)
    return Settings(**base)


class MessageTests(unittest.TestCase):
    def test_qa_requires_question(self):
        with self.assertRaises(PipelineError):
            build_messages("qa", "본문", None)

    def test_extract_prompt_contains_document(self):
        messages = build_messages("extract", "영업소: 서울요금소")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("영업소: 서울요금소", messages[1]["content"])
        self.assertIn("작업: extract", messages[1]["content"])

    def test_parse_fenced_json(self):
        raw = '설명은 무시\n```json\n{"document_type": "통행영수증", "fields": {"영업소": "서울요금소"}}\n```'
        parsed = parse_json_response(raw)
        self.assertEqual(parsed["fields"]["영업소"], "서울요금소")

    def test_parse_invalid_json(self):
        with self.assertRaises(PipelineError):
            parse_json_response("JSON이 아닙니다")


class OfflineTests(unittest.TestCase):
    def test_toll_extract(self):
        markdown = (ROOT / "samples" / "fixtures" / "toll_receipt.md").read_text(encoding="utf-8")
        data = extract_document(markdown)
        self.assertEqual(data["document_type"], "통행영수증")
        self.assertEqual(data["fields"]["통행료"], "3,200원")
        self.assertEqual(data["fields"]["차량번호"], "12가 3456")

    def test_inspection_table(self):
        markdown = (ROOT / "samples" / "fixtures" / "inspection_log.md").read_text(encoding="utf-8")
        data = extract_document(markdown)
        self.assertEqual(data["document_type"], "점검일지")
        self.assertEqual(data["tables"][0]["rows"][1][0], "통신")
        self.assertEqual(data["tables"][0]["rows"][1][2], "주의")

    def test_settlement_total(self):
        markdown = (ROOT / "samples" / "fixtures" / "settlement_table.md").read_text(encoding="utf-8")
        messages = build_messages("qa", markdown, "합계금액은 얼마인가?")
        answer = OfflineChatClient().complete(messages)
        self.assertIn("7,196,500원", answer)

    def test_unknown_question(self):
        markdown = (ROOT / "samples" / "fixtures" / "toll_receipt.md").read_text(encoding="utf-8")
        messages = build_messages("qa", markdown, "날씨는 어떤가?")
        answer = OfflineChatClient().complete(messages)
        self.assertIn("확인되지 않습니다", answer)


class ClientSelectTests(unittest.TestCase):
    def test_auto_offline_when_nothing_configured(self):
        client = build_llm_client("auto", settings(), ollama_up=False)
        self.assertEqual(client.name, "offline")

    def test_auto_openai_when_key_present(self):
        client = build_llm_client("auto", settings(openai_api_key="sk-test"), ollama_up=True)
        self.assertEqual(client.name, "openai")
        self.assertEqual(client.model, "gpt-4o-mini")

    def test_ollama_base_url(self):
        client = build_llm_client("ollama", settings())
        self.assertEqual(client.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(client.model, "qwen2.5:7b")

    def test_openai_requires_key(self):
        with self.assertRaises(LlmConfigError):
            build_llm_client("openai", settings())

    def test_http_client_reads_message(self):
        client = OpenAICompatibleClient(
            name="openai",
            base_url="https://example.invalid/v1",
            api_key="sk-test",
            model="demo",
        )
        payload = {"choices": [{"message": {"content": "서울요금소 통행료는 3,200원입니다."}}]}

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with mock.patch("urllib.request.urlopen", return_value=FakeResponse()) as opener:
            text = client.complete([{"role": "user", "content": "안녕"}])
        self.assertIn("3,200원", text)
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer sk-test")


if __name__ == "__main__":
    unittest.main()
