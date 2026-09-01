"""LLM katmanı: sağlayıcı seçimi (auto), Ollama/OpenAI istemcilerinin hata metinleri ve
düşünen modellerin düşüncesini cevaptan ayıklama. Ağ yok — httpx.Client sahte.

Neden bu kadar mesaj testi: /ask'in ilk kullanıcı deneyimi bir hata metni olacak
("model inmemiş", "Ollama kapalı"); o metin tam komutu söylemezse kimse ikinci kez denemez.
"""

from __future__ import annotations

from typing import ClassVar

import httpx
import pytest

from milvus_rag.config import Settings
from milvus_rag.llm import LLMError, OllamaLLM, OpenAILLM, build_llm, strip_thinking


def _settings(**overrides):
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_auto_provider_follows_the_keys():
    assert _settings().resolved_llm_provider == "ollama"
    assert _settings(anthropic_api_key="sk-ant-x").resolved_llm_provider == "anthropic"
    assert _settings(openai_api_key="sk-x").resolved_llm_provider == "openai"
    both = _settings(anthropic_api_key="a", openai_api_key="o")
    assert both.resolved_llm_provider == "anthropic"  # anahtar varsa Claude öncelikli
    assert _settings(llm_provider="ollama", anthropic_api_key="a").resolved_llm_provider == "ollama"
    assert _settings().resolved_llm_model == _settings(llm_provider="ollama").resolved_llm_model
    assert _settings(llm_model="qwen3.5:4b").resolved_llm_model == "qwen3.5:4b"


def test_build_llm_defaults_to_local_ollama():
    llm = build_llm(_settings())
    assert isinstance(llm, OllamaLLM) and llm.host == "http://localhost:11434"
    assert llm.num_ctx == 16384
    remote = build_llm(_settings(openai_api_key="k", openai_base_url="http://localhost:8000/v1/"))
    assert isinstance(remote, OpenAILLM) and remote.base_url == "http://localhost:8000/v1"


def test_strip_thinking_removes_think_blocks_only():
    assert strip_thinking("<think>uzun düşünce\nsatırlar</think>\nCevap [1].") == "Cevap [1]."
    assert strip_thinking("Düz cevap") == "Düz cevap"


# ---------------------------------------------------------- sahte httpx


class _Response:
    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or (str(payload) if payload is not None else "")

    def json(self):
        return self._payload


class _FakeClient:
    """`httpx.Client` yerine: rota → yanıt. Kayıtlı istekler `calls`'ta."""

    routes: ClassVar[dict[str, list[_Response] | Exception]] = {}
    calls: ClassVar[list[tuple[str, str, dict | None]]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _answer(self, method: str, url: str, json=None):
        _FakeClient.calls.append((method, url, json))
        route = _FakeClient.routes.get(url)
        if isinstance(route, Exception):
            raise route
        if not route:
            raise AssertionError(f"beklenmeyen istek: {method} {url}")
        return route.pop(0) if len(route) > 1 else route[0]

    def get(self, url, **kwargs):
        return self._answer("GET", url, None)

    def post(self, url, json=None, **kwargs):
        return self._answer("POST", url, json)


@pytest.fixture
def fake_http(monkeypatch):
    _FakeClient.routes = {}
    _FakeClient.calls = []
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    return _FakeClient


CHAT = "http://localhost:11434/api/chat"
TAGS = "http://localhost:11434/api/tags"


def test_ollama_disables_thinking_and_retries_for_models_without_it(fake_http):
    llm = OllamaLLM("qwen3.5:9b", "http://localhost:11434/", num_ctx=8192)
    fake_http.routes[CHAT] = [
        _Response(200, {"message": {"content": "<think>hmm</think>\nCevap [1]", "thinking": "x"}})
    ]
    assert llm.complete("sys", "soru") == "Cevap [1]"
    _, _, payload = fake_http.calls[-1]
    assert payload["think"] is False and payload["options"]["num_ctx"] == 8192

    # Düşünmeyen model `think` alanını reddeder → alan olmadan tekrar denenir.
    fake_http.calls.clear()
    fake_http.routes[CHAT] = [
        _Response(400, text='{"error":"model does not support thinking"}'),
        _Response(200, {"message": {"content": "Düz cevap"}}),
    ]
    assert llm.complete("sys", "soru") == "Düz cevap"
    assert "think" in fake_http.calls[0][2] and "think" not in fake_http.calls[1][2]


def test_ollama_errors_tell_the_user_what_to_run(fake_http):
    llm = OllamaLLM("qwen3.5:9b", "http://localhost:11434")
    fake_http.routes[CHAT] = [_Response(404, text='{"error":"model not found"}')]
    with pytest.raises(LLMError, match=r"ollama pull qwen3\.5:9b"):
        llm.complete("sys", "soru")

    fake_http.routes[TAGS] = [_Response(200, {"models": [{"name": "llama3.2:latest"}]})]
    with pytest.raises(LLMError, match=r"indirilmemiş — `ollama pull qwen3\.5:9b`"):
        llm.ping()
    fake_http.routes[TAGS] = [_Response(200, {"models": [{"name": "qwen3.5:9b"}]})]
    assert llm.ping() == "qwen3.5:9b"
    # Etiketsiz model adı ":latest" ile eşleşir.
    fake_http.routes[TAGS] = [_Response(200, {"models": [{"name": "llama3.2:latest"}]})]
    assert OllamaLLM("llama3.2", "http://localhost:11434").ping() == "llama3.2"

    fake_http.routes[TAGS] = httpx.ConnectError("connection refused")
    with pytest.raises(LLMError, match=r"Ollama çalışmıyor .*ollama serve.*ollama\.com/download"):
        llm.ping()


def test_openai_compatible_server_uses_base_url(fake_http):
    llm = OpenAILLM("Qwen/Qwen3.5-9B", "x", "http://localhost:8000/v1")
    fake_http.routes["http://localhost:8000/v1/chat/completions"] = [
        _Response(200, {"choices": [{"message": {"content": "<think>t</think>Cevap"}}]})
    ]
    assert llm.complete("sys", "soru") == "Cevap"
    fake_http.routes["http://localhost:8000/v1/models"] = [_Response(200, {"data": []})]
    assert llm.ping() == "Qwen/Qwen3.5-9B"
