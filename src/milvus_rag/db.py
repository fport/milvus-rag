"""SQLite: repolar, dosya manifesti, işler, webhook tekrarı ve enrichment cache.

Milvus'taki veri türev; "hangi repo hangi commit'te, hangi dosya hangi hash'le
indexli" bilgisi burada. Bir dosyanın manifest satırı YOKSA bir sonraki sync onu
yeni sayar — bu yüzden chunk'ları silmeden önce manifest satırı silinir, yeni
chunk'lar yazıldıktan sonra geri yazılır: yarıda kesilen bir iş eksik bırakmaz.
"""

import json
import sqlite3
import uuid
from collections.abc import Iterable, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from milvus_rag.models import Job, JobStatus, JobTrigger, Repo

_SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    provider        TEXT NOT NULL,
    branch          TEXT NOT NULL,
    local_path      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    owner           TEXT NOT NULL DEFAULT '',
    external_id     TEXT NOT NULL DEFAULT '',
    external_name   TEXT NOT NULL DEFAULT '',
    remote_url      TEXT NOT NULL DEFAULT '',
    web_url         TEXT NOT NULL DEFAULT '',
    last_commit     TEXT NOT NULL DEFAULT '',
    last_indexed_at TEXT NOT NULL DEFAULT '',
    file_count      INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    index_version   TEXT NOT NULL DEFAULT '',
    auto_sync       INTEGER NOT NULL DEFAULT 1,
    error           TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS repos_external_idx ON repos (external_id);

CREATE TABLE IF NOT EXISTS files (
    repo_id     TEXT NOT NULL,
    path        TEXT NOT NULL,
    sha         TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    indexed_at  TEXT NOT NULL,
    PRIMARY KEY (repo_id, path)
);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    repo_id     TEXT NOT NULL,
    trigger     TEXT NOT NULL,
    status      TEXT NOT NULL,
    force       INTEGER NOT NULL DEFAULT 0,
    from_commit TEXT NOT NULL DEFAULT '',
    to_commit   TEXT NOT NULL DEFAULT '',
    stats       TEXT NOT NULL DEFAULT '{}',
    error       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    started_at  TEXT NOT NULL DEFAULT '',
    finished_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS jobs_repo_idx ON jobs (repo_id, created_at);

CREATE TABLE IF NOT EXISTS webhook_events (
    repo_id     TEXT NOT NULL,
    commit_sha  TEXT NOT NULL,
    received_at TEXT NOT NULL,
    PRIMARY KEY (repo_id, commit_sha)
);

CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS enrichment (
    chunk_hash TEXT NOT NULL,
    model      TEXT NOT NULL,
    context    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (chunk_hash, model)
);
"""

_REPO_COLUMNS = (
    "id",
    "name",
    "provider",
    "branch",
    "local_path",
    "status",
    "owner",
    "external_id",
    "external_name",
    "remote_url",
    "web_url",
    "last_commit",
    "last_indexed_at",
    "file_count",
    "chunk_count",
    "index_version",
    "auto_sync",
    "error",
    "created_at",
    "updated_at",
)


def _migrate(connection: sqlite3.Connection) -> None:
    """Şema evrimi. GitHub desteğiyle azure_* kolonları sağlayıcı-bağımsız oldu."""
    table = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'repos'"
    ).fetchone()
    if table is None:
        return
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(repos)")}
    renames = {
        "azure_project": "owner",
        "azure_repo_id": "external_id",
        "azure_repo_name": "external_name",
    }
    for old, new in renames.items():
        if old in columns:
            connection.execute(f"ALTER TABLE repos RENAME COLUMN {old} TO {new}")
    if "azure_repo_id" in columns:
        connection.execute("DROP INDEX IF EXISTS repos_azure_idx")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            _migrate(connection)
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # Bağlantı çağrı başına: worker thread'i ve API aynı dosyayı paylaşır,
        # SQLite bağlantısı ise thread'e bağlıdır.
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    # ------------------------------------------------------------------ repos
    def upsert_repo(self, repo: Repo) -> Repo:
        stamp = now_iso()
        row = {**repo.to_dict(), "created_at": repo.created_at or stamp, "updated_at": stamp}
        row["auto_sync"] = int(row["auto_sync"])
        placeholders = ", ".join(f":{column}" for column in _REPO_COLUMNS)
        updates = ", ".join(
            f"{column} = excluded.{column}" for column in _REPO_COLUMNS if column not in ("id",)
        )
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO repos ({', '.join(_REPO_COLUMNS)}) VALUES ({placeholders})"
                f" ON CONFLICT(id) DO UPDATE SET {updates}",
                row,
            )
        found = self.get_repo(repo.id)
        assert found is not None
        return found

    def update_repo(self, repo_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now_iso()
        if "auto_sync" in fields:
            fields["auto_sync"] = int(bool(fields["auto_sync"]))
        assignments = ", ".join(f"{column} = :{column}" for column in fields)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE repos SET {assignments} WHERE id = :id", {**fields, "id": repo_id}
            )

    def get_repo(self, repo_id: str) -> Repo | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM repos WHERE id = ?", (repo_id,)).fetchone()
        return _to_repo(row) if row else None

    def find_repo_by_external(
        self, external_id: str, branch: str | None = None, provider: str | None = None
    ) -> Repo | None:
        """Webhook'tan gelen kimlikle kayıtlı repoyu bulur (Azure GUID / GitHub id)."""
        query = "SELECT * FROM repos WHERE external_id = ?"
        params: list[Any] = [external_id]
        if provider:
            query += " AND provider = ?"
            params.append(provider)
        if branch:
            query += " AND branch = ?"
            params.append(branch)
        with closing(self._connect()) as connection:
            row = connection.execute(query, params).fetchone()
        return _to_repo(row) if row else None

    def list_repos(self) -> list[Repo]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM repos ORDER BY created_at").fetchall()
        return [_to_repo(row) for row in rows]

    def delete_repo(self, repo_id: str) -> None:
        with self._connect() as connection:
            for table in ("files", "jobs", "webhook_events"):
                connection.execute(f"DELETE FROM {table} WHERE repo_id = ?", (repo_id,))
            connection.execute("DELETE FROM repos WHERE id = ?", (repo_id,))

    # ------------------------------------------------------------------ files
    def manifest(self, repo_id: str) -> dict[str, str]:
        """path → içerik sha256. Artımlı sync'in karşılaştırdığı şey."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT path, sha FROM files WHERE repo_id = ?", (repo_id,)
            ).fetchall()
        return {str(row["path"]): str(row["sha"]) for row in rows}

    def list_files(self, repo_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT path, sha, chunk_count, indexed_at FROM files WHERE repo_id = ?"
                " ORDER BY path",
                (repo_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_file(self, repo_id: str, path: str, sha: str, chunk_count: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO files (repo_id, path, sha, chunk_count, indexed_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(repo_id, path) DO UPDATE SET"
                " sha = excluded.sha, chunk_count = excluded.chunk_count,"
                " indexed_at = excluded.indexed_at",
                (repo_id, path, sha, chunk_count, now_iso()),
            )

    def delete_files(self, repo_id: str, paths: Iterable[str]) -> None:
        paths = list(paths)
        if not paths:
            return
        with self._connect() as connection:
            connection.executemany(
                "DELETE FROM files WHERE repo_id = ? AND path = ?",
                [(repo_id, path) for path in paths],
            )

    def clear_files(self, repo_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))

    # ------------------------------------------------------------------- jobs
    def create_job(self, repo_id: str, trigger: JobTrigger, force: bool = False) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            repo_id=repo_id,
            trigger=trigger,
            status="queued",
            force=force,
            created_at=now_iso(),
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, repo_id, trigger, status, force, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (job.id, job.repo_id, job.trigger, job.status, int(job.force), job.created_at),
            )
        return job

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "stats" in fields:
            fields["stats"] = json.dumps(fields["stats"], ensure_ascii=False)
        if "force" in fields:
            fields["force"] = int(bool(fields["force"]))
        assignments = ", ".join(f"{column} = :{column}" for column in fields)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = :id", {**fields, "id": job_id}
            )

    def get_job(self, job_id: str) -> Job | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _to_job(row) if row else None

    def list_jobs(self, repo_id: str | None = None, limit: int = 50) -> list[Job]:
        query = "SELECT * FROM jobs"
        params: list[Any] = []
        if repo_id:
            query += " WHERE repo_id = ?"
            params.append(repo_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [_to_job(row) for row in rows]

    def active_job(self, repo_id: str) -> Job | None:
        """Kuyrukta bekleyen ya da çalışan iş. Repo başına tek iş garantisi buradan."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE repo_id = ? AND status IN ('queued', 'running')"
                " ORDER BY created_at LIMIT 1",
                (repo_id,),
            ).fetchone()
        return _to_job(row) if row else None

    def queued_jobs(self) -> list[Job]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at"
            ).fetchall()
        return [_to_job(row) for row in rows]

    def fail_running_jobs(self, reason: str) -> int:
        """Süreç yeniden başladığında 'running' kalmış işler yarım kalmıştır."""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status = 'failed', error = ?, finished_at = ?"
                " WHERE status = 'running'",
                (reason, now_iso()),
            )
        return int(cursor.rowcount)

    # ---------------------------------------------------------------- webhook
    def record_webhook(self, repo_id: str, commit_sha: str) -> bool:
        """Aynı (repo, commit) ikinci kez gelirse False: Azure yeniden dener, biz iki iş açmayız."""
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO webhook_events (repo_id, commit_sha, received_at)"
                " VALUES (?, ?, ?)",
                (repo_id, commit_sha, now_iso()),
            )
        return cursor.rowcount == 1

    # ------------------------------------------------------------ ayarlar
    def get_app_settings(self, keys: Sequence[str]) -> dict[str, str]:
        """UI'dan kaydedilen ayarlar (kimlik bilgileri dahil). Env'i ezerler."""
        if not keys:
            return {}
        marks = ", ".join("?" for _ in keys)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT key, value FROM app_settings WHERE key IN ({marks})", list(keys)
            ).fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def set_app_setting(self, key: str, value: str | None) -> None:
        """None ya da boş değer kaydı siler → env'deki değere geri düşülür."""
        with self._connect() as connection:
            if value:
                connection.execute(
                    "INSERT OR REPLACE INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, value, now_iso()),
                )
            else:
                connection.execute("DELETE FROM app_settings WHERE key = ?", (key,))

    # ------------------------------------------------------------- enrichment
    def get_enrichments(self, chunk_hashes: Sequence[str], model: str) -> dict[str, str]:
        if not chunk_hashes:
            return {}
        found: dict[str, str] = {}
        with closing(self._connect()) as connection:
            for start in range(0, len(chunk_hashes), 500):
                batch = chunk_hashes[start : start + 500]
                marks = ", ".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT chunk_hash, context FROM enrichment"
                    f" WHERE model = ? AND chunk_hash IN ({marks})",
                    [model, *batch],
                ).fetchall()
                found.update({str(row["chunk_hash"]): str(row["context"]) for row in rows})
        return found

    def set_enrichments(self, items: Iterable[tuple[str, str]], model: str) -> None:
        stamp = now_iso()
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO enrichment (chunk_hash, model, context, created_at)"
                " VALUES (?, ?, ?, ?)",
                [(chunk_hash, model, context, stamp) for chunk_hash, context in items],
            )


def _to_repo(row: sqlite3.Row) -> Repo:
    return Repo(
        id=str(row["id"]),
        name=str(row["name"]),
        provider=row["provider"],
        branch=str(row["branch"]),
        local_path=str(row["local_path"]),
        status=row["status"],
        owner=str(row["owner"]),
        external_id=str(row["external_id"]),
        external_name=str(row["external_name"]),
        remote_url=str(row["remote_url"]),
        web_url=str(row["web_url"]),
        last_commit=str(row["last_commit"]),
        last_indexed_at=str(row["last_indexed_at"]),
        file_count=int(row["file_count"]),
        chunk_count=int(row["chunk_count"]),
        index_version=str(row["index_version"]),
        auto_sync=bool(row["auto_sync"]),
        error=str(row["error"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _to_job(row: sqlite3.Row) -> Job:
    status: JobStatus = row["status"]
    return Job(
        id=str(row["id"]),
        repo_id=str(row["repo_id"]),
        trigger=row["trigger"],
        status=status,
        force=bool(row["force"]),
        from_commit=str(row["from_commit"]),
        to_commit=str(row["to_commit"]),
        stats=json.loads(row["stats"] or "{}"),
        error=str(row["error"]),
        created_at=str(row["created_at"]),
        started_at=str(row["started_at"]),
        finished_at=str(row["finished_at"]),
    )
