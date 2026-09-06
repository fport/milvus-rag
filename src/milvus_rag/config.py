"""Typed settings. Every value has a default that works; even an empty environment
comes up with Milvus and the local models.

Variables prefixed `RAG_*` belong to this service. Credentials (AZURE_DEVOPS_*,
ANTHROPIC_API_KEY, OPENAI_API_KEY) have no prefix, so they share the names every
other tool uses.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# When the chunker or the embedding changes, old vectors no longer match new queries.
# This version string is stored per repo; if it differs, the next sync forces a full
# re-index.
CHUNKER_VERSION = "ast-v1"

# The default model per provider. The Ollama choice was measured (README → "Local LLM");
# override it with RAG_LLM_MODEL.
_DEFAULT_LLM_MODEL: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o",
    "ollama": "qwen3.5:9b",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Servis --------------------------------------------------------------
    port: int = 8090
    log_level: str = "INFO"
    data_dir: Path = Path("data")

    # --- Milvus --------------------------------------------------------------
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "code_chunks"

    # --- Embedding -----------------------------------------------------------
    embedding_backend: Literal["local", "openai"] = "local"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)
    # BGE-M3 takes 8192 tokens but chunks never exceed 2000 bytes; a short window = speed.
    embedding_max_tokens: int = Field(default=1024, gt=0)
    embedding_device: str | None = None

    # --- Retrieval -----------------------------------------------------------
    # The defaults were measured (evals/results/, sample-api, 42 questions, k=8):
    # dense 0.786/0.678 (recall/MRR) · hybrid 0.786/0.604 · bm25 1.0/1.0 on symbols.
    # bge-reranker-v2-m3 HURT on this corpus (0.762/0.508, TR 0.684→0.579) and pushed
    # p50 to 2-4.4 s → off. Measure again with `rag eval` before turning it on.
    search_mode: Literal["auto", "dense", "bm25", "hybrid"] = "auto"
    prose_mode: Literal["dense", "hybrid"] = "dense"
    rrf_k: int = Field(default=60, gt=0)
    top_k: int = Field(default=8, gt=0)
    candidates: int = Field(default=40, gt=0)
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_tokens: int = Field(default=1024, gt=0)
    cache_ttl_seconds: int = Field(default=300, ge=0)
    cache_size: int = Field(default=1000, ge=0)
    # A weak-match SIGNAL, not a filter: if the best dense score is below this, the
    # response carries `weak_match=True`; the results still come back and the decision
    # belongs to the consumer (agent / LLM). Measured (2026-09-01, sample-api, 38 golden
    # prose questions + 12 irrelevant ones): the top-1 dense of the golden hits had a
    # median of 0.636 / min 0.526, the irrelevant ones a median of 0.531 / max 0.598.
    # The distributions overlap → no hard threshold is possible; 0.55 flags 10/12 of the
    # irrelevant ones and wrongly flags 3/29 real ones. Cosine is not calibrated:
    # re-measure when the embedding model changes.
    weak_dense_score: float = Field(default=0.55, ge=0.0, le=1.0)
    # A hard FLOOR (CRAG's lower band): a chunk scoring below it never comes back — so
    # "have you been to a five-star resort" does not get a 0.37 blog chunk. Measured
    # (2026-09-01): no answer found in the golden set fell below 0.526, and the best of
    # the irrelevant queries scored 0.37-0.60. 0.45 only cuts the absurd tail; the grey
    # zone (0.45-0.55) is flagged with `weak_match` and still returned. 0 = off.
    # Not applied to BM25.
    min_dense_score: float = Field(default=0.45, ge=0.0, le=1.0)

    # --- Chunking ------------------------------------------------------------
    chunk_max_bytes: int = Field(default=2000, gt=0)
    chunk_min_bytes: int = Field(default=200, ge=0)
    max_file_bytes: int = Field(default=512_000, gt=0)
    # Extra extensions, comma-separated (like ".proto,.graphql").
    extra_extensions: str = ""

    # --- LLM -----------------------------------------------------------------
    # "auto": Claude if an Anthropic key exists, else OpenAI if an OpenAI key exists, else
    # local Ollama — so `git clone` + `ollama pull` is enough for /ask, with zero setup.
    llm_provider: Literal["auto", "anthropic", "openai", "ollama"] = "auto"
    llm_model: str | None = None
    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY")
    )
    # Any OpenAI-compatible server: vLLM, LM Studio, llama.cpp server → change the base URL.
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_host: str = "http://localhost:11434"
    # Ollama's default context is 4k; an 8-chunk prompt is ~6k tokens → it truncated silently.
    ollama_num_ctx: int = Field(default=16384, gt=0)
    enrich_enabled: bool = False
    enrich_batch_chunks: int = Field(default=6, gt=0)
    # The language chunk descriptions are written in. The measured gain comes from adding
    # prose in the language questions are asked in, so match your question traffic.
    enrich_language: str = "English"

    # --- Azure DevOps --------------------------------------------------------
    azure_org_url: str | None = Field(
        default=None, validation_alias=AliasChoices("AZURE_DEVOPS_ORG_URL")
    )
    azure_pat: str | None = Field(default=None, validation_alias=AliasChoices("AZURE_DEVOPS_PAT"))

    # --- GitHub ----------------------------------------------------------
    # The token is optional: public repos clone without one; private repos need it.
    github_token: str | None = Field(default=None, validation_alias=AliasChoices("GITHUB_TOKEN"))
    github_api_url: str = "https://api.github.com"

    # --- MCP -----------------------------------------------------------------
    # `/mcp` sits behind DNS-rebinding protection: localhost only by default. If the
    # service will be reached over MCP from another address (rag.company.local, say),
    # add the host here — comma-separated, port included ("rag.company.local,10.0.0.5:8090").
    mcp_allowed_hosts: str = ""
    # --- Refresh -------------------------------------------------------------
    # --- Tazeleme ------------------------------------------------------------
    webhook_secret: str | None = None
    poll_interval_seconds: int = Field(default=300, ge=0)

    @model_validator(mode="after")
    def _min_below_max(self) -> "Settings":
        if self.chunk_min_bytes >= self.chunk_max_bytes:
            msg = (
                f"chunk_min_bytes ({self.chunk_min_bytes}) must be smaller than "
                f"chunk_max_bytes ({self.chunk_max_bytes})"
            )
            raise ValueError(msg)
        if self.min_dense_score > self.weak_dense_score:
            msg = (
                f"min_dense_score ({self.min_dense_score}) cannot exceed weak_dense_score "
                f"({self.weak_dense_score}): the floor stays below the weak-match note"
            )
            raise ValueError(msg)
        return self

    # --- Derived -------------------------------------------------------------
    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rag.db"

    @property
    def resolved_llm_provider(self) -> Literal["anthropic", "openai", "ollama"]:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "ollama"

    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or _DEFAULT_LLM_MODEL[self.resolved_llm_provider]

    @property
    def index_version(self) -> str:
        """Everything that shapes the index. If it changes, re-index fully."""
        return (
            f"{CHUNKER_VERSION};embed={self.embedding_backend}:{self.embedding_model}"
            f";dim={self.embedding_dim};chunk={self.chunk_min_bytes}-{self.chunk_max_bytes}"
            f";enrich={int(self.enrich_enabled)}"
        )

    @property
    def azure_configured(self) -> bool:
        return bool(self.azure_org_url and self.azure_pat)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
