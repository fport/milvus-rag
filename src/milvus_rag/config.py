"""Tipli ayarlar. Her değerin çalışan bir varsayılanı var; boş bir ortam bile
Milvus + yerel modellerle ayağa kalkar.

`RAG_*` önekli değişkenler bu servise özel. Kimlik bilgileri (AZURE_DEVOPS_*,
ANTHROPIC_API_KEY, OPENAI_API_KEY) öneksiz — api/ tarafıyla aynı isimleri
paylaşsınlar diye.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Chunker ya da embedding değişince eski vektörler yeni sorgularla uyuşmaz.
# Bu sürüm string'i repo başına saklanır; farklıysa bir sonraki sync zorla tam
# yeniden indexler.
CHUNKER_VERSION = "ast-v1"

_DEFAULT_LLM_MODEL: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o",
    "ollama": "qwen2.5:7b",
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
    # BGE-M3 8192 token alır ama chunk'lar 2000 byte'ı geçmez; kısa pencere = hız.
    embedding_max_tokens: int = Field(default=1024, gt=0)
    embedding_device: str | None = None

    # --- Retrieval -----------------------------------------------------------
    # Varsayılanlar ölçüldü (evals/results/, sample-api, 42 soru, k=8):
    # dense 0.786/0.678 (recall/MRR) · hybrid 0.786/0.604 · bm25 sembollerde 1.0/1.0.
    # bge-reranker-v2-m3 bu korpusta ZARAR etti (0.762/0.508, TR 0.684→0.579) ve
    # p50'yi 2-4.4 sn yaptı → kapalı. Açmadan önce `rag eval` ile yeniden ölç.
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

    # --- Chunking ------------------------------------------------------------
    chunk_max_bytes: int = Field(default=2000, gt=0)
    chunk_min_bytes: int = Field(default=200, ge=0)
    max_file_bytes: int = Field(default=512_000, gt=0)
    # Virgülle ayrılmış ek uzantılar (".proto,.graphql" gibi).
    extra_extensions: str = ""

    # --- LLM -----------------------------------------------------------------
    llm_provider: Literal["anthropic", "openai", "ollama"] = "anthropic"
    llm_model: str | None = None
    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY")
    )
    ollama_host: str = "http://localhost:11434"
    enrich_enabled: bool = False
    enrich_batch_chunks: int = Field(default=6, gt=0)

    # --- Azure DevOps --------------------------------------------------------
    azure_org_url: str | None = Field(
        default=None, validation_alias=AliasChoices("AZURE_DEVOPS_ORG_URL")
    )
    azure_pat: str | None = Field(default=None, validation_alias=AliasChoices("AZURE_DEVOPS_PAT"))

    # --- GitHub ----------------------------------------------------------
    # Token isteğe bağlı: public repo tokensız klonlanır; private için gerekli.
    github_token: str | None = Field(default=None, validation_alias=AliasChoices("GITHUB_TOKEN"))
    github_api_url: str = "https://api.github.com"

    # --- MCP -----------------------------------------------------------------
    # `/mcp` DNS rebinding koruması altında: varsayılan yalnızca localhost. Servise
    # başka bir adresten (rag.sirket.local gibi) MCP ile bağlanılacaksa host'u buraya
    # ekle — virgülle ayrılmış, port dahil ("rag.sirket.local,10.0.0.5:8090").
    mcp_allowed_hosts: str = ""

    # --- Tazeleme ------------------------------------------------------------
    webhook_secret: str | None = None
    poll_interval_seconds: int = Field(default=300, ge=0)

    @model_validator(mode="after")
    def _min_below_max(self) -> "Settings":
        if self.chunk_min_bytes >= self.chunk_max_bytes:
            msg = (
                f"chunk_min_bytes ({self.chunk_min_bytes}) chunk_max_bytes'tan "
                f"({self.chunk_max_bytes}) küçük olmalı"
            )
            raise ValueError(msg)
        return self

    # --- Türetilen -----------------------------------------------------------
    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rag.db"

    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or _DEFAULT_LLM_MODEL[self.llm_provider]

    @property
    def index_version(self) -> str:
        """Index'in şeklini belirleyen her şey. Değişirse tam yeniden index."""
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
