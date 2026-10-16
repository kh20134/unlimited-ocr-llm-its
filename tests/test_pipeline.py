import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr_llm_its.config import Settings
from ocr_llm_its.errors import PipelineError
from ocr_llm_its.pipeline import run_document


def settings() -> Settings:
    return Settings(
        hf_token=None,
        openai_api_base="https://api.openai.com/v1",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="qwen2.5:7b",
        ocr_backend="auto",
        llm_backend="auto",
        model_name="baidu/Unlimited-OCR",
        fixtures_dir=ROOT / "samples" / "fixtures",
    )


class FakeLLM:
    name = "mock"
    model = "mock-model"

    def __init__(self, reply: str):
        self.reply = reply
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.reply


class PipelineTests(unittest.TestCase):
    def test_fixture_summarize_offline(self):
        result = run_document(
            ROOT / "samples" / "toll_receipt.png",
            "summarize",
            ocr_backend="fixture",
            llm_backend="offline",
            settings=settings(),
            ollama_up=False,
        )
        self.assertEqual(result.ocr_backend, "fixture")
        self.assertEqual(result.llm_backend, "offline")
        self.assertIn("서울요금소", result.result)
        self.assertIn("3,200원", result.result)
        self.assertIn("픽스처", result.notice)

    def test_extract_uses_mocked_llm_json(self):
        reply = json.dumps(
            {"document_type": "통행영수증", "title": "통행료 영수증", "fields": {"영업소": "서울요금소"}},
            ensure_ascii=False,
        )
        fake = FakeLLM("```json\n" + reply + "\n```")
        result = run_document(
            ROOT / "samples" / "toll_receipt.png",
            "extract",
            ocr_backend="fixture",
            settings=settings(),
            llm_client=fake,
        )
        self.assertEqual(result.llm_backend, "mock")
        self.assertEqual(result.result["fields"]["영업소"], "서울요금소")
        self.assertIn("서울요금소", fake.messages[-1]["content"])

    def test_qa_mocked(self):
        fake = FakeLLM("통행료는 3,200원입니다.")
        result = run_document(
            ROOT / "samples" / "inspection_log.png",
            "qa",
            "통신 점검 결과는?",
            ocr_backend="fixture",
            settings=settings(),
            llm_client=fake,
        )
        self.assertIn("3,200원", result.result)
        self.assertIn("통신 점검 결과는?", fake.messages[-1]["content"])

    def test_qa_without_question(self):
        with self.assertRaises(PipelineError):
            run_document(
                ROOT / "samples" / "toll_receipt.png",
                "qa",
                ocr_backend="fixture",
                llm_backend="offline",
                settings=settings(),
            )

    def test_auto_ocr_and_llm_without_gpu_or_key(self):
        with mock.patch("ocr_llm_its.ocr.select.cuda_available", return_value=False):
            result = run_document(
                ROOT / "samples" / "settlement_table.pdf",
                "extract",
                ocr_backend="auto",
                llm_backend="auto",
                settings=settings(),
                ollama_up=False,
            )
        self.assertEqual(result.ocr_backend, "fixture")
        self.assertEqual(result.llm_backend, "offline")
        self.assertEqual(result.result["fields"]["합계금액"], "7,196,500원")
        self.assertEqual(result.result["tables"][0]["rows"][0][0], "1종")

    def test_assets_exist(self):
        png = ROOT / "docs" / "images" / "architecture.png"
        seq = ROOT / "docs" / "images" / "sequence.png"
        self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))
        self.assertGreater(png.stat().st_size, 10_000)
        self.assertTrue(seq.read_bytes().startswith(b"\x89PNG"))
        self.assertTrue((ROOT / "samples" / "toll_receipt.png").read_bytes().startswith(b"\x89PNG"))
        self.assertTrue((ROOT / "samples" / "inspection_log.png").read_bytes().startswith(b"\x89PNG"))
        self.assertTrue((ROOT / "samples" / "settlement_table.pdf").read_bytes().startswith(b"%PDF"))


class CliTests(unittest.TestCase):
    def test_cli_extract_and_qa(self):
        extract = subprocess.run(
            [
                sys.executable,
                "-m",
                "ocr_llm_its",
                "run",
                "--input",
                "samples/toll_receipt.png",
                "--task",
                "extract",
                "--backend",
                "fixture",
                "--llm",
                "offline",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(extract.returncode, 0, extract.stderr)
        payload = json.loads(extract.stdout)
        self.assertEqual(payload["result"]["fields"]["영업소"], "서울요금소")

        qa = subprocess.run(
            [
                sys.executable,
                "-m",
                "ocr_llm_its",
                "run",
                "--input",
                "samples/inspection_log.png",
                "--task",
                "qa",
                "--question",
                "통신 점검 결과는?",
                "--backend",
                "fixture",
                "--llm",
                "offline",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(qa.returncode, 0, qa.stderr)
        answer = json.loads(qa.stdout)["result"]
        self.assertIn("주의", answer)
        self.assertIn("함체 재부팅", answer)

    def test_cli_qa_missing_question(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ocr_llm_its",
                "run",
                "--input",
                "samples/toll_receipt.png",
                "--task",
                "qa",
                "--llm",
                "offline",
                "--backend",
                "fixture",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("question", proc.stderr)


if __name__ == "__main__":
    unittest.main()
