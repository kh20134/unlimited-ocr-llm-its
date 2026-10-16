from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Directory that contains ``samples/`` when running from a checkout."""
    packaged = Path(__file__).resolve().parents[1]
    if (packaged / "samples").is_dir():
        return packaged
    cwd = Path.cwd()
    if (cwd / "samples").is_dir():
        return cwd
    return packaged


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines into the environment without overriding existing keys."""
    env_path = path or (Path.cwd() / ".env")
    if not env_path.is_file():
        fallback = repo_root() / ".env"
        env_path = fallback if fallback.is_file() else env_path
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Settings:
    hf_token: str | None
    openai_api_base: str
    openai_api_key: str | None
    openai_model: str
    ollama_host: str
    ollama_model: str
    ocr_backend: str
    llm_backend: str
    model_name: str
    fixtures_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        root = repo_root()
        fixtures = os.environ.get("FIXTURES_DIR", "").strip()
        token = os.environ.get("HF_TOKEN", "").strip() or None
        api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None
        return cls(
            hf_token=token,
            openai_api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1").strip()
            or "https://api.openai.com/v1",
            openai_api_key=api_key,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
            ollama_host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
            or "http://127.0.0.1:11434",
            ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b").strip() or "qwen2.5:7b",
            ocr_backend=os.environ.get("OCR_BACKEND", "auto").strip() or "auto",
            llm_backend=os.environ.get("LLM_BACKEND", "auto").strip() or "auto",
            model_name=os.environ.get("OCR_MODEL", "baidu/Unlimited-OCR").strip()
            or "baidu/Unlimited-OCR",
            fixtures_dir=Path(fixtures) if fixtures else root / "samples" / "fixtures",
        )
