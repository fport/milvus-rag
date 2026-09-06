"""LLM providers: for the /ask answer and the optional chunk descriptions.

One interface: `complete(system, user) -> str`. Anthropic through the official SDK
(claude-opus-5, server-side refusal fallback on), OpenAI and Ollama through a thin
httpx client. Retrieval does not depend on this layer; with no LLM, /search still
works and /ask returns 503.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from milvus_rag.config import Settings
from milvus_rag.log import get_logger

log = get_logger("llm")

Effort = str  # low | medium | high | xhigh | max


class LLMError(RuntimeError):
    pass


# Thinking models (qwen3, deepseek-r1, gpt-oss) return their reasoning in a separate
# field or in a <think> block; keep it out of the answer.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_thinking(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


class LLM(ABC):
    provider: str
    model: str

    @abstractmethod
    def complete(
        self, system: str, user: str, max_tokens: int = 4096, effort: Effort | None = None
    ) -> str: ...

    def ping(self) -> str:
        """Verifies the key with a cheap request; an error → LLMError. Default: no check."""
        return self.model


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        import anthropic

        self.model = model
        # api_key None → the SDK resolves it from the environment (ANTHROPIC_API_KEY or a profile).
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def ping(self) -> str:
        import anthropic

        try:
            # Listing models: spends no tokens and reports a wrong key with a 401.
            self.client.models.list(limit=1)
        except anthropic.APIStatusError as error:
            msg = f"Anthropic {error.status_code}: {error.message}"
            raise LLMError(msg) from error
        except anthropic.AnthropicError as error:
            msg = f"could not reach Anthropic: {error}"
            raise LLMError(msg) from error
        return self.model

    def complete(
        self, system: str, user: str, max_tokens: int = 4096, effort: Effort | None = None
    ) -> str:
        import anthropic

        kwargs: dict[str, Any] = {}
        if effort:
            kwargs["output_config"] = {"effort": effort}
        try:
            # Streaming + get_final_message: a long answer never trips the HTTP timeout.
            # fallbacks="default": if a safety classifier refuses the request, the server
            # retries the same request on a suitable fallback model.
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **kwargs,
            ) as stream:
                message = stream.get_final_message()
        except anthropic.RateLimitError as error:
            msg = f"Anthropic rate limit: {error.message}"
            raise LLMError(msg) from error
        except anthropic.APIStatusError as error:
            msg = f"Anthropic {error.status_code}: {error.message}"
            raise LLMError(msg) from error
        except anthropic.APIConnectionError as error:
            msg = f"could not reach Anthropic: {error}"
            raise LLMError(msg) from error
        except anthropic.AnthropicError as error:
            # Pre-request failures land here: no credential, or one that could not resolve.
            msg = f"Anthropic client: {error}"
            raise LLMError(msg) from error

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            msg = f"the model refused the request (category: {category})"
            raise LLMError(msg)
        return "".join(block.text for block in message.content if block.type == "text")


class OpenAILLM(LLM):
    """OpenAI and OpenAI-compatible servers (vLLM, LM Studio, llama.cpp): base_url changes."""

    provider = "openai"

    def __init__(
        self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1"
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def ping(self) -> str:
        import httpx

        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/models", headers=self._headers())
        except httpx.HTTPError as error:
            msg = f"{self.base_url} is unreachable: {error}"
            raise LLMError(msg) from error
        if response.status_code >= 400:
            msg = f"OpenAI {response.status_code}: {response.text[:200]}"
            raise LLMError(msg)
        return self.model

    def complete(
        self, system: str, user: str, max_tokens: int = 4096, effort: Effort | None = None
    ) -> str:
        import httpx

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
        }
        try:
            with httpx.Client(timeout=180.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions", headers=self._headers(), json=payload
                )
        except httpx.HTTPError as error:
            msg = f"{self.base_url} is unreachable: {error}"
            raise LLMError(msg) from error
        if response.status_code >= 400:
            msg = f"OpenAI {response.status_code}: {response.text[:300]}"
            raise LLMError(msg)
        choices = response.json().get("choices") or []
        if not choices:
            msg = "OpenAI returned an empty response"
            raise LLMError(msg)
        return strip_thinking(str(choices[0]["message"].get("content") or ""))


class OllamaLLM(LLM):
    """Yerel model. Kurulum: https://ollama.com/download → `ollama pull <model>`."""

    provider = "ollama"

    def __init__(self, model: str, host: str, num_ctx: int = 16384) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx

    def _unreachable(self, error: Exception) -> LLMError:
        return LLMError(
            f"Ollama is not running ({self.host}): {error}. Open the app or run `ollama serve`; "
            "install: https://ollama.com/download"
        )

    def ping(self) -> str:
        """Is the server up, is the model pulled — and if not, print the exact command."""
        import httpx

        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{self.host}/api/tags")
        except httpx.HTTPError as error:
            raise self._unreachable(error) from error
        if response.status_code >= 400:
            msg = f"Ollama {response.status_code}: {response.text[:200]}"
            raise LLMError(msg)
        names = {str(item.get("name", "")) for item in response.json().get("models") or []}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names:
            msg = f"`{self.model}` is not pulled in Ollama — `ollama pull {self.model}`"
            raise LLMError(msg)
        return self.model

    def complete(
        self, system: str, user: str, max_tokens: int = 4096, effort: Effort | None = None
    ) -> str:
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Thinking off: 10x faster, and a cited answer does not need a chain of thought.
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2,
                "num_ctx": self.num_ctx,
            },
        }
        try:
            with httpx.Client(timeout=600.0) as client:
                response = client.post(f"{self.host}/api/chat", json=payload)
                if response.status_code == 400 and "think" in response.text.lower():
                    # A non-thinking model does not know the `think` field: retry without it.
                    retry = {key: value for key, value in payload.items() if key != "think"}
                    response = client.post(f"{self.host}/api/chat", json=retry)
        except httpx.HTTPError as error:
            raise self._unreachable(error) from error
        if response.status_code == 404:
            msg = f"`{self.model}` is not pulled in Ollama — `ollama pull {self.model}`"
            raise LLMError(msg)
        if response.status_code >= 400:
            msg = f"Ollama {response.status_code}: {response.text[:300]}"
            raise LLMError(msg)
        content = str((response.json().get("message") or {}).get("content") or "")
        return strip_thinking(content)


def build_llm(settings: Settings) -> LLM | None:
    """None when nothing is configured: retrieval works without an LLM.

    With provider `auto` it is resolved from the keys (see Settings.resolved_llm_provider);
    the Ollama client is always built — if the server is down, /ask gives a clear error.
    """
    provider = settings.resolved_llm_provider
    model = settings.resolved_llm_model
    if provider == "anthropic":
        try:
            return AnthropicLLM(model, settings.anthropic_api_key)
        except Exception as error:
            log.warning("could not build the Anthropic client; /ask is off", error=str(error))
            return None
    if provider == "openai":
        if not settings.openai_api_key:
            log.warning("no OPENAI_API_KEY; /ask is off")
            return None
        return OpenAILLM(model, settings.openai_api_key, settings.openai_base_url)
    return OllamaLLM(model, settings.ollama_host, settings.ollama_num_ctx)
