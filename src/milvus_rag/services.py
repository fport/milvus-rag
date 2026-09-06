"""Builds the pieces from the settings. The CLI and the API take the same stack from
here, so there is no third way of wiring it up."""

from __future__ import annotations

from dataclasses import dataclass

from milvus_rag.config import Settings, get_settings
from milvus_rag.db import Database
from milvus_rag.index.embed import Embedder, build_embedder
from milvus_rag.index.enrich import Enricher
from milvus_rag.index.pipeline import Indexer
from milvus_rag.index.store import MilvusStore
from milvus_rag.jobs import JobRunner
from milvus_rag.llm import LLM, build_llm
from milvus_rag.log import get_logger, setup_logging
from milvus_rag.repos import RepoService
from milvus_rag.search.rerank import CrossEncoderReranker
from milvus_rag.search.retrieve import Retriever
from milvus_rag.sources.azure import AzureDevOps
from milvus_rag.sources.github import GitHub

log = get_logger("services")

# Secrets that can be entered from the UI. Each overrides its env counterpart;
# an empty record falls back to env.
CREDENTIAL_KEYS = (
    "github_token",
    "azure_org_url",
    "azure_pat",
    "webhook_secret",
    "anthropic_api_key",
)


def resolve_credentials(settings: Settings, db: Database) -> dict[str, str | None]:
    """A value saved from the UI overrides the env; deleting the record falls back to env."""
    saved = db.get_app_settings(CREDENTIAL_KEYS)
    return {
        "github_token": saved.get("github_token") or settings.github_token,
        "azure_org_url": saved.get("azure_org_url") or settings.azure_org_url,
        "azure_pat": saved.get("azure_pat") or settings.azure_pat,
        "webhook_secret": saved.get("webhook_secret") or settings.webhook_secret,
        "anthropic_api_key": saved.get("anthropic_api_key") or settings.anthropic_api_key,
    }


def credential_sources(settings: Settings, db: Database) -> dict[str, str | None]:
    """Where each secret comes from: "ui", "env" or None. The value itself is never returned."""
    saved = db.get_app_settings(CREDENTIAL_KEYS)
    env = {
        "github_token": settings.github_token,
        "azure_org_url": settings.azure_org_url,
        "azure_pat": settings.azure_pat,
        "webhook_secret": settings.webhook_secret,
        "anthropic_api_key": settings.anthropic_api_key,
    }
    return {
        key: "ui" if saved.get(key) else ("env" if env[key] else None) for key in CREDENTIAL_KEYS
    }


def _build_llm_and_enricher(
    settings: Settings, db: Database, anthropic_api_key: str | None
) -> tuple[LLM | None, Enricher | None]:
    # Settings are frozen; copy it to override the key. The LLM lives only in the answer layer.
    llm = build_llm(settings.model_copy(update={"anthropic_api_key": anthropic_api_key}))
    enricher = (
        Enricher(llm, db, settings.enrich_batch_chunks, settings.enrich_language)
        if settings.enrich_enabled and llm
        else None
    )
    if settings.enrich_enabled and llm is None:
        log.warning("RAG_ENRICH_ENABLED is on but there is no LLM; enrichment will be skipped")
    return llm, enricher


def apply_credentials(services: Services) -> None:
    """Rebuilds the clients with the current credentials — without restarting the process."""
    creds = resolve_credentials(services.settings, services.db)
    if services.github is not None:
        services.github.close()
    if services.azure is not None:
        services.azure.close()
    services.github = GitHub(creds["github_token"], services.settings.github_api_url)
    services.azure = (
        AzureDevOps(creds["azure_org_url"] or "", creds["azure_pat"] or "")
        if creds["azure_org_url"] and creds["azure_pat"]
        else None
    )
    services.webhook_secret = creds["webhook_secret"]
    services.llm, services.enricher = _build_llm_and_enricher(
        services.settings, services.db, creds["anthropic_api_key"]
    )
    # Everyone holding the same client object should see the new one.
    for holder in (services.indexer, services.repos, services.jobs):
        holder.github = services.github
        holder.azure = services.azure
    services.indexer.enricher = services.enricher


@dataclass(slots=True)
class Services:
    settings: Settings
    db: Database
    store: MilvusStore
    embedder: Embedder
    reranker: CrossEncoderReranker | None
    retriever: Retriever
    llm: LLM | None
    enricher: Enricher | None
    azure: AzureDevOps | None
    github: GitHub | None
    indexer: Indexer
    repos: RepoService
    jobs: JobRunner
    # The resolved webhook secret (UI > env); the webhook endpoints read this, not settings.
    webhook_secret: str | None = None

    def warm_up(self) -> None:
        """Load the models at startup; do not put 20 seconds on the first request."""
        self.embedder.warm_up()
        if self.reranker is not None:
            self.reranker.warm_up()

    def close(self) -> None:
        if self.azure is not None:
            self.azure.close()
        if self.github is not None:
            self.github.close()


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.repos_dir.mkdir(parents=True, exist_ok=True)

    db = Database(settings.db_path)
    store = MilvusStore(settings.milvus_uri, settings.milvus_collection, settings.embedding_dim)
    embedder = build_embedder(settings)
    reranker = (
        CrossEncoderReranker(settings.rerank_model, settings.rerank_max_tokens)
        if settings.rerank_enabled
        else None
    )
    retriever = Retriever(settings, store, embedder, reranker)
    creds = resolve_credentials(settings, db)
    llm, enricher = _build_llm_and_enricher(settings, db, creds["anthropic_api_key"])
    azure = (
        AzureDevOps(creds["azure_org_url"] or "", creds["azure_pat"] or "")
        if creds["azure_org_url"] and creds["azure_pat"]
        else None
    )
    github = GitHub(creds["github_token"], settings.github_api_url)
    indexer = Indexer(settings, db, store, embedder, enricher, azure, github)
    repos = RepoService(settings, db, store, azure, github)
    jobs = JobRunner(settings, db, indexer, retriever, azure, github)
    return Services(
        settings=settings,
        db=db,
        store=store,
        embedder=embedder,
        reranker=reranker,
        retriever=retriever,
        llm=llm,
        enricher=enricher,
        azure=azure,
        github=github,
        indexer=indexer,
        repos=repos,
        jobs=jobs,
        webhook_secret=creds["webhook_secret"],
    )
