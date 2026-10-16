from __future__ import annotations

import json
import urllib.error
import urllib.request

from ocr_llm_its.config import Settings
from ocr_llm_its.errors import LlmConfigError, PipelineError
from ocr_llm_its.llm.offline import OfflineChatClient


class OpenAICompatibleClient:
    """Chat client for OpenAI and any server that implements ``/v1/chat/completions``.

    Ollama's OpenAI-compatible endpoint is the same shape, so one client covers
    both. The API key for a local Ollama server can be any non-empty placeholder.
    """

    def __init__(self, name: str, base_url: str, api_key: str, model: str, timeout: float = 120):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]]) -> str:
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": messages,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise PipelineError(f"LLM 요청이 실패했습니다 ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise PipelineError(f"LLM 서버에 연결하지 못했습니다: {exc.reason}") from exc
        try:
            body = json.loads(raw)
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            raise PipelineError("LLM 응답에서 메시지 본문을 찾지 못했습니다.") from exc


def ollama_reachable(host: str, timeout: float = 0.4) -> bool:
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except Exception:
        return False


def build_llm_client(choice: str, settings: Settings, *, ollama_up: bool | None = None):
    """Select the chat backend.

    ``auto`` prefers ``OPENAI_API_KEY``, then a reachable Ollama host, then the
    offline demo client. Explicit ``openai`` or ``ollama`` never silently
    switches to offline.
    """
    selected = (choice or "auto").strip().lower()
    if selected not in {"auto", "openai", "ollama", "offline"}:
        raise PipelineError("LLM 백엔드는 auto, openai, ollama, offline 중 하나여야 합니다.")
    if selected == "offline":
        return OfflineChatClient()
    if selected == "openai":
        return _openai_client(settings)
    if selected == "ollama":
        return _ollama_client(settings)
    if settings.openai_api_key:
        return _openai_client(settings)
    up = ollama_reachable(settings.ollama_host) if ollama_up is None else ollama_up
    if up:
        return _ollama_client(settings)
    return OfflineChatClient()


def _openai_client(settings: Settings) -> OpenAICompatibleClient:
    if not settings.openai_api_key:
        raise LlmConfigError(
            "OPENAI_API_KEY가 없습니다. .env에 키를 넣거나 --llm ollama 또는 --llm offline 을 사용하세요."
        )
    base = settings.openai_api_base.rstrip("/")
    return OpenAICompatibleClient(
        name="openai",
        base_url=base,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )


def _ollama_client(settings: Settings) -> OpenAICompatibleClient:
    if not settings.ollama_host:
        raise LlmConfigError("OLLAMA_HOST가 비어 있습니다.")
    host = settings.ollama_host.rstrip("/")
    if host.endswith("/v1"):
        base = host
    else:
        base = host + "/v1"
    return OpenAICompatibleClient(
        name="ollama",
        base_url=base,
        api_key="ollama",
        model=settings.ollama_model,
    )


def llm_notice(client) -> str:
    if getattr(client, "name", "") == "offline":
        return (
            "OPENAI_API_KEY가 없고 도달 가능한 Ollama도 없어 오프라인 데모 정리기를 사용합니다. "
            "이것은 신경망 LLM이 아니라 마크다운을 JSON, 짧은 한국어 요약, 질의응답으로 바꾸는 규칙입니다. "
            "실제 모델은 .env의 OPENAI_API_KEY 또는 OLLAMA_HOST로 연결하세요."
        )
    model = getattr(client, "model", "")
    return f"LLM 백엔드 {client.name} 모델 {model} 을 사용합니다."
