"""Indexes one repo: refresh the source → compare the manifest → write what changed.

So that every commit does not re-embed the whole repo, the sha256 of each file's
content is stored (SQLite `files`). For a changed file the chunks at that path are
deleted first and re-inserted; a deleted file is only deleted. The result: a push pays
the embedding cost of the files it touched, and nothing else.

Resilience to interruption: a file's manifest row is removed BEFORE its chunks are
deleted, and put back AFTER the new chunks are written. If the process dies in the
middle the file is absent from the manifest → the next sync counts it as new.

Change detection is based on the content hash, not on a commit diff: there are no git
edge cases from renames, mode changes or submodules, and local (non-git) directories
take the same path.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from milvus_rag.config import Settings
from milvus_rag.db import Database, now_iso
from milvus_rag.index.chunk import ChunkerConfig, build_indexed_text, chunk_file
from milvus_rag.index.embed import Embedder
from milvus_rag.index.enrich import Enricher
from milvus_rag.index.scrub import scrub
from milvus_rag.index.store import MilvusStore
from milvus_rag.log import get_logger
from milvus_rag.models import Repo, SourceFile
from milvus_rag.sources import files as source_files
from milvus_rag.sources import git
from milvus_rag.sources.azure import AzureDevOps
from milvus_rag.sources.github import GitHub

log = get_logger("index")

# Files are embedded and inserted once this many chunks have piled up. Embedding is far
# faster in a batch, and this keeps the memory ceiling bounded.
FLUSH_CHUNKS = 96


@dataclass(slots=True)
class IndexStats:
    commit: str = ""
    files_seen: int = 0
    added: int = 0
    modified: int = 0
    deleted: int = 0
    unchanged: int = 0
    skipped: int = 0
    chunks_written: int = 0
    chunks_deleted_paths: int = 0
    redactions: int = 0
    enriched: int = 0
    enrich_cached: int = 0
    enrich_rejected: int = 0
    embed_seconds: float = 0.0
    total_seconds: float = 0.0
    forced: bool = False
    reason: str = ""
    progress: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["embed_seconds"] = round(self.embed_seconds, 1)
        data["total_seconds"] = round(self.total_seconds, 1)
        return data


@dataclass(slots=True)
class _Pending:
    """A file whose chunks are waiting in the buffer."""

    path: str
    sha: str
    chunk_count: int


class Indexer:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        store: MilvusStore,
        embedder: Embedder,
        enricher: Enricher | None = None,
        azure: AzureDevOps | None = None,
        github: GitHub | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.store = store
        self.embedder = embedder
        self.enricher = enricher
        self.azure = azure
        self.github = github
        self.chunker = ChunkerConfig(settings.chunk_max_bytes, settings.chunk_min_bytes)
        self.extra_extensions = source_files.parse_extra_extensions(settings.extra_extensions)

    # ------------------------------------------------------------- source
    def refresh_source(self, repo: Repo) -> str:
        """Brings the working copy in line with the remote; returns the known commit."""
        root = Path(repo.local_path)
        if repo.provider == "local":
            if not root.is_dir():
                msg = f"no such local directory: {root}"
                raise FileNotFoundError(msg)
            return git.head_commit(root) if git.is_work_tree(root) else ""

        auth_header = None
        if repo.provider == "azure":
            if self.azure is None:
                msg = "Azure DevOps is not configured (AZURE_DEVOPS_ORG_URL / AZURE_DEVOPS_PAT)"
                raise RuntimeError(msg)
            auth_header = self.azure.git_auth_header()
        elif repo.provider == "github" and self.github is not None:
            # Returns None when there is no token; a public repo clones anonymously.
            auth_header = self.github.git_auth_header()

        if not (root / ".git").exists():
            log.info("cloning", repo=repo.id, branch=repo.branch)
            git.clone(repo.remote_url, root, repo.branch, auth_header)
            return git.head_commit(root)
        return git.fetch_and_reset(root, repo.branch, auth_header)

    # -------------------------------------------------------------- index
    def sync(
        self,
        repo: Repo,
        force: bool = False,
        on_progress: Callable[[IndexStats], None] | None = None,
    ) -> IndexStats:
        started = time.perf_counter()
        stats = IndexStats()
        root = Path(repo.local_path)

        stats.commit = self.refresh_source(repo)

        if repo.index_version and repo.index_version != self.settings.index_version:
            force = True
            stats.reason = "the index version changed"
        stats.forced = force

        manifest = {} if force else self.db.manifest(repo.id)
        if force:
            # A full re-index: the old chunks and the manifest go in one shot.
            self.db.clear_files(repo.id)
            self.store.delete_repo(repo.id)

        seen: set[str] = set()
        changed: list[SourceFile] = []
        for path in source_files.list_candidate_paths(root):
            if not source_files.is_indexable_path(path, self.extra_extensions):
                continue
            source = source_files.read_source_file(root, path, self.settings.max_file_bytes)
            if source is None:
                stats.skipped += 1
                continue
            seen.add(path)
            stats.files_seen += 1
            previous = manifest.get(path)
            if previous == source.sha:
                stats.unchanged += 1
                continue
            if previous is None:
                stats.added += 1
            else:
                stats.modified += 1
            changed.append(source)

        deleted = sorted(set(manifest) - seen)
        stats.deleted = len(deleted)
        if deleted:
            self.db.delete_files(repo.id, deleted)
            self.store.delete_paths(repo.id, deleted)
            stats.chunks_deleted_paths += len(deleted)

        self._write_files(repo, changed, stats, on_progress)

        if self.enricher is not None:
            stats.enriched = self.enricher.generated
            stats.enrich_cached = self.enricher.cached
            stats.enrich_rejected = self.enricher.rejected

        stats.total_seconds = time.perf_counter() - started
        self.db.update_repo(
            repo.id,
            status="ready",
            error="",
            last_commit=stats.commit,
            last_indexed_at=now_iso(),
            file_count=stats.files_seen,
            chunk_count=self.store.count(repo.id),
            index_version=self.settings.index_version,
        )
        log.info(
            "sync bitti",
            repo=repo.id,
            files=stats.files_seen,
            added=stats.added,
            modified=stats.modified,
            deleted=stats.deleted,
            unchanged=stats.unchanged,
            chunks=stats.chunks_written,
            seconds=round(stats.total_seconds, 1),
        )
        return stats

    def _write_files(
        self,
        repo: Repo,
        sources: list[SourceFile],
        stats: IndexStats,
        on_progress: Callable[[IndexStats], None] | None,
    ) -> None:
        rows: list[dict[str, Any]] = []
        texts: list[str] = []
        pending: list[_Pending] = []
        total = len(sources)
        done = 0
        last_report = time.perf_counter()

        def flush() -> None:
            nonlocal rows, texts, pending
            if not pending:
                return
            paths = [item.path for item in pending]
            # The manifest goes first (so an interruption counts the file as "new"), then
            # the old chunks, then the new ones; the manifest is written back last.
            self.db.delete_files(repo.id, paths)
            self.store.delete_paths(repo.id, paths)
            if rows:
                mark = time.perf_counter()
                vectors = self.embedder.encode(texts)
                stats.embed_seconds += time.perf_counter() - mark
                for row, vector in zip(rows, vectors, strict=True):
                    row["dense"] = vector.tolist()
                stats.chunks_written += self.store.insert(rows)
            for item in pending:
                self.db.set_file(repo.id, item.path, item.sha, item.chunk_count)
            rows, texts, pending = [], [], []

        for source in sources:
            cleaned = scrub(source.text)
            stats.redactions += cleaned.total
            records = chunk_file(source.path, cleaned.text, self.chunker)
            contexts = (
                self.enricher.describe(source.path, cleaned.text, records)
                if self.enricher is not None and source.category == "code"
                else [""] * len(records)
            )
            stamp = now_iso()
            for record, context in zip(records, contexts, strict=True):
                indexed_text = build_indexed_text(record, context)
                rows.append(
                    {
                        "id": f"{repo.id}:{source.path}#{record.ordinal}",
                        "repo_id": repo.id,
                        "path": source.path,
                        "symbol": record.symbol[:500],
                        "parent_symbol": record.parent_symbol[:500],
                        "kind": record.kind[:32],
                        "lang": source.lang,
                        "category": source.category,
                        "ordinal": record.ordinal,
                        "start_line": record.start_line,
                        "end_line": record.end_line,
                        "content": record.text,
                        "indexed_text": indexed_text,
                        "context": context,
                        "symbols": " ".join(record.symbols),
                        "file_sha": source.sha,
                        "chunk_hash": hashlib.sha256(record.text.encode()).hexdigest(),
                        "indexed_at": stamp,
                    }
                )
                texts.append(indexed_text)
            pending.append(_Pending(source.path, source.sha, len(records)))
            done += 1
            if len(rows) >= FLUSH_CHUNKS:
                flush()
            if on_progress is not None and time.perf_counter() - last_report > 5:
                stats.progress = {"files_done": done, "files_total": total}
                on_progress(stats)
                last_report = time.perf_counter()
        flush()
        stats.progress = {"files_done": done, "files_total": total}
