"""Types that travel between modules. If something crosses a boundary it is
defined here; no bare dicts travel."""

from dataclasses import dataclass, field
from typing import Any, Literal

Provider = Literal["azure", "github", "local", "git"]
RepoStatus = Literal["pending", "indexing", "ready", "error"]
JobStatus = Literal["queued", "running", "done", "failed", "skipped"]
JobTrigger = Literal["manual", "webhook", "poll", "startup"]
Category = Literal["code", "doc", "other"]


@dataclass(frozen=True, slots=True)
class Repo:
    id: str
    name: str
    provider: Provider
    branch: str
    local_path: str
    status: RepoStatus = "pending"
    # owner: the Azure project or the GitHub owner; external_id/name: the provider's id.
    owner: str = ""
    external_id: str = ""
    external_name: str = ""
    remote_url: str = ""
    web_url: str = ""
    last_commit: str = ""
    last_indexed_at: str = ""
    file_count: int = 0
    chunk_count: int = 0
    index_version: str = ""
    auto_sync: bool = True
    error: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "branch": self.branch,
            "status": self.status,
            "owner": self.owner,
            "external_id": self.external_id,
            "external_name": self.external_name,
            "remote_url": self.remote_url,
            "web_url": self.web_url,
            "local_path": self.local_path,
            "last_commit": self.last_commit,
            "last_indexed_at": self.last_indexed_at,
            "file_count": self.file_count,
            "chunk_count": self.chunk_count,
            "index_version": self.index_version,
            "auto_sync": self.auto_sync,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    repo_id: str
    trigger: JobTrigger
    status: JobStatus
    force: bool = False
    from_commit: str = ""
    to_commit: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "trigger": self.trigger,
            "status": self.status,
            "force": self.force,
            "from_commit": self.from_commit,
            "to_commit": self.to_commit,
            "stats": self.stats,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A single file read from a repo."""

    path: str
    text: str
    sha: str
    lang: str
    category: Category


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A piece produced by the chunker, not embedded yet.

    `text` is the clean content that goes to the LLM; `header` is the
    "what am I, where do I live" lines prepended for embedding and BM25. They are
    kept apart so that a citation shows the real file text.
    """

    ordinal: int
    text: str
    start_line: int
    end_line: int
    symbol: str
    parent_symbol: str
    kind: str
    header: str
    symbols: tuple[str, ...] = ()


@dataclass(slots=True)
class Hit:
    """A search result. Scores are carried per channel (dense / bm25 / rrf /
    rerank) — which channel found what is visible in the data, not only in the UI."""

    id: str
    repo_id: str
    path: str
    symbol: str
    parent_symbol: str
    kind: str
    lang: str
    category: str
    start_line: int
    end_line: int
    content: str
    context: str
    scores: dict[str, float] = field(default_factory=dict)
    rank: int = 0

    @property
    def ref(self) -> str:
        """The form golden sets expect: path::symbol."""
        return f"{self.path}::{self.symbol}" if self.symbol else self.path

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "path": self.path,
            "symbol": self.symbol,
            "parent_symbol": self.parent_symbol,
            "kind": self.kind,
            "lang": self.lang,
            "category": self.category,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "context": self.context,
            "scores": {key: round(value, 6) for key, value in self.scores.items()},
            "rank": self.rank,
        }
