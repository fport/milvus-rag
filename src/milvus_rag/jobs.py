"""The job queue: index jobs run in order, on a single worker thread.

Why a single worker: indexing is CPU-bound (embedding) and writes to Milvus; indexing
two repos at once slows both down and speeds up neither. There is a one-pending-job-
per-repo guarantee (like jobId dedupe in BullMQ): if five pushes hit the same repo,
one job runs and one waits.

Job records live in SQLite; after a restart the ones left "running" become failed and
the "queued" ones are re-queued.

The poller: if a webhook was never set up, or one was missed, it compares the branch
head upstream against the last indexed commit.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from milvus_rag.config import Settings
from milvus_rag.db import Database, now_iso
from milvus_rag.index.pipeline import Indexer, IndexStats
from milvus_rag.log import get_logger
from milvus_rag.models import Job, JobTrigger
from milvus_rag.search.retrieve import Retriever
from milvus_rag.sources.azure import AzureDevOps, AzureError
from milvus_rag.sources.github import GitHub, GitHubError

log = get_logger("jobs")


class JobRunner:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        indexer: Indexer,
        retriever: Retriever,
        azure: AzureDevOps | None,
        github: GitHub | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.indexer = indexer
        self.retriever = retriever
        self.azure = azure
        self.github = github
        self._queue: asyncio.Queue[str] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-index")
        self._tasks: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    # ---------------------------------------------------------------- queue
    def enqueue(self, repo_id: str, trigger: JobTrigger, force: bool = False) -> tuple[Job, bool]:
        """(job, is_new). If a job is already pending, that one comes back."""
        active = [
            job
            for job in self.db.list_jobs(repo_id, limit=20)
            if job.status in ("queued", "running")
        ]
        queued = next((job for job in active if job.status == "queued"), None)
        if queued is not None:
            if force and not queued.force:
                self.db.update_job(queued.id, force=True)
                queued = self.db.get_job(queued.id) or queued
            return queued, False
        job = self.db.create_job(repo_id, trigger, force)
        self._submit(job.id)
        log.info("job queued", job=job.id, repo=repo_id, trigger=trigger, force=force)
        return job, True

    def _submit(self, job_id: str) -> None:
        if self._queue is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, job_id)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        interrupted = self.db.fail_running_jobs("the process restarted; the job was cut short")
        if interrupted:
            log.warning("interrupted jobs marked failed", count=interrupted)
        for job in self.db.queued_jobs():
            self._queue.put_nowait(job.id)
        self._tasks.append(asyncio.create_task(self._worker(), name="rag-job-worker"))
        if self.settings.poll_interval_seconds > 0 and (self.azure or self.github):
            self._tasks.append(asyncio.create_task(self._poller(), name="rag-poller"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _worker(self) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        while True:
            job_id = await self._queue.get()
            try:
                await loop.run_in_executor(self._executor, self.run_job, job_id)
            except Exception as error:
                log.error("job crashed unexpectedly", job=job_id, error=str(error))
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ run
    def run_job(self, job_id: str) -> IndexStats | None:
        """Runs one job on this thread. Both the worker and the CLI use it."""
        job = self.db.get_job(job_id)
        if job is None or job.status != "queued":
            return None
        repo = self.db.get_repo(job.repo_id)
        if repo is None:
            self.db.update_job(
                job_id, status="skipped", error="repo was deleted", finished_at=now_iso()
            )
            return None

        self.db.update_job(
            job_id, status="running", started_at=now_iso(), from_commit=repo.last_commit
        )
        self.db.update_repo(repo.id, status="indexing", error="")
        started = time.perf_counter()

        def report(stats: IndexStats) -> None:
            self.db.update_job(job_id, stats=stats.to_dict())

        try:
            stats = self.indexer.sync(repo, force=job.force, on_progress=report)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            log.error("sync failed", job=job_id, repo=repo.id, error=message)
            self.db.update_job(job_id, status="failed", error=message[:2000], finished_at=now_iso())
            self.db.update_repo(repo.id, status="error", error=message[:2000])
            return None

        self.db.update_job(
            job_id,
            status="done",
            to_commit=stats.commit,
            stats=stats.to_dict(),
            finished_at=now_iso(),
        )
        self.retriever.invalidate()
        log.info(
            "job finished",
            job=job_id,
            repo=repo.id,
            seconds=round(time.perf_counter() - started, 1),
            chunks=stats.chunks_written,
        )
        return stats

    def run_now(self, repo_id: str, trigger: JobTrigger = "manual", force: bool = False) -> Job:
        """For the CLI: run it on this thread without queueing and return the job record."""
        job = self.db.create_job(repo_id, trigger, force)
        self.run_job(job.id)
        return self.db.get_job(job.id) or job

    # --------------------------------------------------------------- poller
    async def _poller(self) -> None:
        interval = self.settings.poll_interval_seconds
        await asyncio.sleep(min(interval, 30))
        while True:
            try:
                await asyncio.get_running_loop().run_in_executor(None, self.poll_once)
            except Exception as error:
                log.warning("poll round failed", error=str(error))
            await asyncio.sleep(interval)

    def poll_once(self) -> list[dict[str, Any]]:
        """Compare the heads of the remote repos; queue the ones that moved."""
        triggered: list[dict[str, Any]] = []
        for repo in self.db.list_repos():
            if not repo.auto_sync or repo.status == "indexing":
                continue
            try:
                if repo.provider == "azure" and self.azure is not None:
                    head = self.azure.branch_head(repo.owner, repo.external_id, repo.branch)
                elif repo.provider == "github" and self.github is not None:
                    head = self.github.branch_head(repo.owner, repo.external_name, repo.branch)
                else:
                    continue
            except (AzureError, GitHubError) as error:
                log.warning("poll: could not read head", repo=repo.id, error=str(error))
                continue
            if head and head != repo.last_commit:
                job, created = self.enqueue(repo.id, "poll")
                triggered.append(
                    {"repo_id": repo.id, "head": head, "job_id": job.id, "new": created}
                )
                log.info("poll: change found", repo=repo.id, head=head[:12], job=job.id)
        return triggered
