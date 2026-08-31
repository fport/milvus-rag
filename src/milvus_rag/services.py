"""Parçaları ayarlardan kurar. CLI ve API aynı yığını buradan alır; üç yerde
üç farklı kablolama olmasın."""

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

    def warm_up(self) -> None:
        """Modelleri açılışta yükle; ilk isteğe 20 saniye bindirme."""
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
    llm = build_llm(settings)
    enricher = (
        Enricher(llm, db, settings.enrich_batch_chunks) if settings.enrich_enabled and llm else None
    )
    if settings.enrich_enabled and llm is None:
        log.warning("RAG_ENRICH_ENABLED açık ama LLM yok; enrichment atlanacak")
    azure = (
        AzureDevOps(settings.azure_org_url or "", settings.azure_pat or "")
        if settings.azure_configured
        else None
    )
    github = GitHub(settings.github_token, settings.github_api_url)
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
    )
