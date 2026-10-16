import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr_llm_its.config import Settings
from ocr_llm_its.errors import CudaRequiredError, FixtureNotFoundError
from ocr_llm_its.ocr.clean import clean_ocr_markdown, remove_det
from ocr_llm_its.ocr.fixture import FixtureOcr
from ocr_llm_its.ocr.local import LocalUnlimitedOcr
from ocr_llm_its.ocr.select import build_ocr_backend


def settings() -> Settings:
    return Settings(
        hf_token=None,
        openai_api_base="https://api.openai.com/v1",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="qwen2.5:7b",
        ocr_backend="auto",
        llm_backend="offline",
        model_name="baidu/Unlimited-OCR",
        fixtures_dir=ROOT / "samples" / "fixtures",
    )


class CleanTests(unittest.TestCase):
    def test_remove_det_groups_blocks(self):
        raw = "\n".join(
            [
                "<|det|>title [0, 0, 10, 10]<|/det|># 통행료 영수증",
                "<|det|>text [0, 20, 10, 30]<|/det|>영업소: 서울요금소",
                "<|det|>image [0, 40, 10, 50]<|/det|>그림",
                "이어지는 줄",
            ]
        )
        text = remove_det(raw)
        self.assertIn("# 통행료 영수증", text)
        self.assertIn("영업소: 서울요금소", text)
        self.assertNotIn("그림", text)
        self.assertNotIn("<|det|>", text)
        self.assertIn("이어지는 줄", text)

    def test_clean_strips_stop_token_without_markers(self):
        raw = "차로: 3<｜end▁of▁sentence｜>"
        self.assertEqual(clean_ocr_markdown(raw), "차로: 3")


class FixtureTests(unittest.TestCase):
    def test_reads_matching_stem(self):
        result = FixtureOcr(ROOT / "samples" / "fixtures").parse(ROOT / "samples" / "toll_receipt.png")
        self.assertEqual(result.backend, "fixture")
        self.assertIn("서울요금소", result.markdown)
        self.assertTrue(result.source.endswith("toll_receipt.md"))

    def test_pdf_stem_matches_fixture(self):
        result = FixtureOcr(ROOT / "samples" / "fixtures").parse(ROOT / "samples" / "settlement_table.pdf")
        self.assertIn("7,196,500원", result.markdown)

    def test_missing_fixture(self):
        with self.assertRaises(FixtureNotFoundError):
            FixtureOcr(ROOT / "samples" / "fixtures").parse(ROOT / "samples" / "missing.png")


class BackendSelectTests(unittest.TestCase):
    def test_auto_falls_back_without_cuda(self):
        with mock.patch("ocr_llm_its.ocr.select.cuda_available", return_value=False):
            backend = build_ocr_backend("auto", settings())
        self.assertEqual(backend.name, "fixture")

    def test_auto_uses_local_when_cuda(self):
        with mock.patch("ocr_llm_its.ocr.select.cuda_available", return_value=True):
            backend = build_ocr_backend("auto", settings())
        self.assertIsInstance(backend, LocalUnlimitedOcr)
        self.assertEqual(backend.name, "local")

    def test_local_refuses_without_cuda(self):
        with mock.patch("ocr_llm_its.ocr.select.cuda_available", return_value=False):
            with self.assertRaises(CudaRequiredError):
                build_ocr_backend("local", settings())

    def test_import_does_not_require_torch(self):
        import ocr_llm_its.ocr.local as local
        import ocr_llm_its.pipeline  # noqa: F401

        self.assertTrue(hasattr(local, "LocalUnlimitedOcr"))


if __name__ == "__main__":
    unittest.main()
