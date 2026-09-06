"""The HTTP interface.

    uv run rag serve            # or: uv run uvicorn milvus_rag.api:app --port 8090

The models are loaded once in the lifespan. Index jobs are queued and run on a single
background worker; /repos/{id} and /jobs/{id} show the progress.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from milvus_rag import __version__
from milvus_rag.config import get_settings
from milvus_rag.llm import LLMError
from milvus_rag.log import get_logger
from milvus_rag.mcp_server import MCP_PATH, McpSlashMiddleware, build_mcp_server
from milvus_rag.repos import RepoError
from milvus_rag.search.answer import ask as run_ask
from milvus_rag.search.retrieve import SearchRequest
from milvus_rag.services import (
    CREDENTIAL_KEYS,
    Services,
    apply_credentials,
    build_services,
    credential_sources,
)
from milvus_rag.sources.azure import AzureError
from milvus_rag.sources.files import FileReadError, read_indexed_slice
from milvus_rag.sources.github import GitHubError
from milvus_rag.webhooks import (
    PushEvent,
    parse_azure_push,
    parse_github_push,
    verify_github_signature,
    verify_secret,
)

log = get_logger("api")

# The MCP transport's DNS-rebinding guard: the default localhost set.
_LOCAL_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOCAL_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


# ------------------------------------------------------------------- schemas


class AddRepoBody(BaseModel):
    provider: Literal["azure", "github", "local", "git"] = "azure"
    # azure / github
    project: str | None = None
    repo: str | None = Field(
        default=None, description="Azure: repo name or GUID · GitHub: owner/repo"
    )
    branch: str | None = None
    # local
    path: str | None = None
    # git
    url: str | None = None
    name: str | None = None
    auto_sync: bool = True
    index_now: bool = True


class SyncBody(BaseModel):
    force: bool = False


class CredentialsBody(BaseModel):
    """A field that is not sent is left alone; an empty string deletes the record (back to env)."""

    github_token: str | None = None
    azure_org_url: str | None = None
    azure_pat: str | None = None
    webhook_secret: str | None = None
    anthropic_api_key: str | None = None
    verify: bool = True


class SearchBody(BaseModel):
    q: str = Field(min_length=1, max_length=2000)
    repo_ids: list[str] = Field(default_factory=list)
    k: int | None = Field(default=None, ge=1, le=50)
    mode: Literal["auto", "dense", "bm25", "hybrid"] | None = None
    rerank: bool | None = None
    candidates: int | None = Field(default=None, ge=1, le=200)
    path_prefix: str = ""
    lang: str = ""
    category: Literal["code", "doc", "other"] | None = None

    def to_request(self) -> SearchRequest:
        return SearchRequest(
            query=self.q,
            repo_ids=tuple(self.repo_ids),
            k=self.k,
            mode=self.mode,
            rerank=self.rerank,
            candidates=self.candidates,
            path_prefix=self.path_prefix,
            lang=self.lang,
            category=self.category or "",
        )


# ------------------------------------------------------------------- uygulama


def create_app(services: Services | None = None, warm_up: bool = True) -> FastAPI:
    # The MCP server runs in the same process as the app, over the same retriever.
    # Because the services are born in the lifespan, the tools resolve them lazily.
    mcp = build_mcp_server(lambda: app.state.services)
    mcp_settings = (services.settings if services else get_settings()).mcp_allowed_hosts
    extra_hosts = [host.strip() for host in mcp_settings.split(",") if host.strip()]
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(
            allowed_hosts=[*_LOCAL_HOSTS, *extra_hosts],
            allowed_origins=[
                *_LOCAL_ORIGINS,
                *(f"http://{h}" for h in extra_hosts),
                *(f"https://{h}" for h in extra_hosts),
            ],
        )
        if extra_hosts
        else None,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        built = services or build_services()
        app.state.services = built
        if warm_up:
            built.warm_up()
        if not built.store.healthy():
            log.warning(
                "Milvus is unreachable; /search will come back empty",
                uri=built.settings.milvus_uri,
            )
        await built.jobs.start()
        log.info(
            "ready",
            port=built.settings.port,
            collection=built.settings.milvus_collection,
            embedding=built.embedder.name,
            rerank=built.reranker.model_name if built.reranker else "off",
            azure="on" if built.azure else "off",
            llm=built.llm.model if built.llm else "off",
            mcp="/mcp",
        )
        # The session manager runs from here, not from the mounted sub-app's own lifespan
        # (FastAPI does not call a mounted app's lifespan).
        async with mcp.session_manager.run():
            try:
                yield
            finally:
                await built.jobs.stop()
                built.close()

    app = FastAPI(title="Milvus RAG", version=__version__, lifespan=lifespan)
    app.mount(MCP_PATH, mcp_app)
    app.add_middleware(McpSlashMiddleware)

    def svc(request: Request) -> Services:
        return request.app.state.services  # type: ignore[no-any-return]

    S = Annotated[Services, Depends(svc)]  # noqa: N806

    @app.exception_handler(RepoError)
    async def _repo_error(_: Request, error: RepoError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.exception_handler(AzureError)
    async def _azure_error(_: Request, error: AzureError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.exception_handler(GitHubError)
    async def _github_error(_: Request, error: GitHubError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(error)})

    # ------------------------------------------------------------------- UI
    static_dir = Path(__file__).parent / "static"

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    # ---------------------------------------------------------------- health
    @app.get("/health")
    def health(s: S) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "milvus": s.store.healthy(),
            "repos": len(s.db.list_repos()),
            "embedding": s.embedder.name,
            "rerank": s.reranker.model_name if s.reranker else None,
            "llm": s.llm.model if s.llm else None,
            "azure": s.azure is not None,
            "github_token": bool(s.github and s.github.token),
            "webhook_secret_set": bool(s.webhook_secret),
            "poll_interval_seconds": s.settings.poll_interval_seconds,
        }

    # ------------------------------------------------------------- kimlikler
    @app.get("/settings/credentials")
    def get_credentials(s: S, probe: bool = Query(False)) -> dict[str, Any]:
        """Masked status: where each credential comes from. The secrets never come back.

        `probe=1`: a live ping to the LLM (is Ollama up / is the model pulled / is the key
        valid). The UI asks for it; tests and scripts do not — no network.
        """
        source = credential_sources(s.settings, s.db)
        return {
            "github": {
                "token_set": bool(s.github and s.github.token),
                "source": source["github_token"],
            },
            "azure": {
                "configured": s.azure is not None,
                "org_url": s.azure.org_url if s.azure else (s.settings.azure_org_url or ""),
                "source": source["azure_pat"],
            },
            "webhook": {"secret_set": bool(s.webhook_secret), "source": source["webhook_secret"]},
            "llm": _llm_view(s, source["anthropic_api_key"], probe),
        }

    @app.put("/settings/credentials")
    def put_credentials(body: CredentialsBody, s: S) -> dict[str, Any]:
        """Save the credentials, rebuild the clients live, verify them if asked.

        The secrets sit in data/rag.db in plain text — the service is meant to live on an
        internal network anyway (see DEPLOYMENT.md). A field sent as an empty string is deleted.
        """
        touched = {key for key in CREDENTIAL_KEYS if key in body.model_fields_set}
        for key in touched:
            value = getattr(body, key)
            s.db.set_app_setting(key, (value or "").strip() or None)
        apply_credentials(s)

        # Only touched credentials are verified: changing the webhook secret should not call GitHub.
        result: dict[str, Any] = {"github": None, "azure": None, "llm": None}
        if body.verify:
            if "github_token" in touched and s.github and s.github.token:
                try:
                    result["github"] = {"ok": True, "login": s.github.whoami()}
                except GitHubError as error:
                    result["github"] = {"ok": False, "detail": str(error)}
            if touched & {"azure_org_url", "azure_pat"} and s.azure is not None:
                try:
                    projects = s.azure.list_projects()
                    result["azure"] = {"ok": True, "projects": len(projects)}
                except AzureError as error:
                    result["azure"] = {"ok": False, "detail": str(error)}
            if "anthropic_api_key" in touched and s.llm is not None:
                try:
                    result["llm"] = {"ok": True, "model": s.llm.ping()}
                except LLMError as error:
                    result["llm"] = {"ok": False, "detail": str(error)}
        return result

    # ----------------------------------------------------------------- azure
    @app.get("/azure/projects")
    def azure_projects(s: S) -> list[dict[str, Any]]:
        if s.azure is None:
            raise HTTPException(503, "Azure DevOps is not configured")
        return [project.to_dict() for project in s.azure.list_projects()]

    @app.get("/azure/projects/{project}/repos")
    def azure_repos(project: str, s: S) -> list[dict[str, Any]]:
        if s.azure is None:
            raise HTTPException(503, "Azure DevOps is not configured")
        registered = {
            repo.external_id: repo.id
            for repo in s.db.list_repos()
            if repo.provider == "azure" and repo.external_id
        }
        return [
            {**repo.to_dict(), "registered_as": registered.get(repo.id)}
            for repo in s.azure.list_repos(project)
        ]

    @app.get("/github/{owner}/repos")
    def github_repos(owner: str, s: S) -> list[dict[str, Any]]:
        if s.github is None:
            raise HTTPException(503, "the GitHub client is not configured")
        registered = {
            repo.external_id: repo.id
            for repo in s.db.list_repos()
            if repo.provider == "github" and repo.external_id
        }
        return [
            {**repo.to_dict(), "registered_as": registered.get(repo.id)}
            for repo in s.github.list_repos(owner)
        ]

    # ----------------------------------------------------------------- repos
    @app.get("/repos")
    def list_repos(s: S) -> list[dict[str, Any]]:
        return [_repo_view(s, repo.id) for repo in s.db.list_repos()]

    @app.post("/repos", status_code=201)
    def add_repo(body: AddRepoBody, s: S) -> dict[str, Any]:
        if body.provider == "azure":
            if not body.project or not body.repo:
                raise HTTPException(422, "azure needs project and repo")
            repo = s.repos.register_azure(body.project, body.repo, body.branch, body.auto_sync)
        elif body.provider == "github":
            if not body.repo:
                raise HTTPException(422, "github needs repo ('owner/repo')")
            repo = s.repos.register_github(body.repo, body.branch, body.auto_sync)
        elif body.provider == "local":
            if not body.path:
                raise HTTPException(422, "local needs path")
            repo = s.repos.register_local(body.path, body.name, body.auto_sync)
        else:
            if not body.url:
                raise HTTPException(422, "git needs url")
            repo = s.repos.register_git(body.url, body.branch or "main", body.name, body.auto_sync)
        job = None
        if body.index_now:
            job, _ = s.jobs.enqueue(repo.id, "manual")
        return {**_repo_view(s, repo.id), "job": job.to_dict() if job else None}

    @app.get("/repos/{repo_id}")
    def get_repo(repo_id: str, s: S) -> dict[str, Any]:
        if s.db.get_repo(repo_id) is None:
            raise HTTPException(404, "repo yok")
        return _repo_view(s, repo_id)

    @app.delete("/repos/{repo_id}")
    def delete_repo(repo_id: str, s: S) -> dict[str, Any]:
        s.repos.remove(repo_id)
        s.retriever.invalidate()
        return {"deleted": repo_id}

    @app.post("/repos/{repo_id}/sync", status_code=202)
    def sync_repo(repo_id: str, body: SyncBody, s: S) -> dict[str, Any]:
        if s.db.get_repo(repo_id) is None:
            raise HTTPException(404, "repo yok")
        job, created = s.jobs.enqueue(repo_id, "manual", force=body.force)
        return {"job": job.to_dict(), "created": created}

    @app.get("/repos/{repo_id}/files")
    def repo_files(repo_id: str, s: S) -> list[dict[str, Any]]:
        if s.db.get_repo(repo_id) is None:
            raise HTTPException(404, "repo yok")
        return s.db.list_files(repo_id)

    @app.get("/repos/{repo_id}/jobs")
    def repo_jobs(repo_id: str, s: S, limit: int = Query(20, ge=1, le=200)) -> list[dict[str, Any]]:
        return [job.to_dict() for job in s.db.list_jobs(repo_id, limit)]

    @app.get("/repos/{repo_id}/file")
    def read_file(
        repo_id: str,
        s: S,
        path: str = Query(min_length=1),
        start: int = Query(1, ge=1),
        end: int | None = Query(None, ge=1),
    ) -> dict[str, Any]:
        """A line range from a file — for an agent's `read_code` tool.

        Only indexed files are readable (the manifest); `stale` says the content on disk
        changed after the last index. The reasoning lives in sources/files.py.
        """
        repo = s.db.get_repo(repo_id)
        if repo is None:
            raise HTTPException(404, "repo yok")
        try:
            piece = read_indexed_slice(
                Path(repo.local_path), s.db.manifest(repo_id), path, start, end, max_lines=1000
            )
        except FileReadError as error:
            raise HTTPException(404, str(error)) from error
        return {
            "repo_id": repo_id,
            "path": piece.path,
            "start": piece.start,
            "end": piece.end,
            "total_lines": piece.total_lines,
            "text": piece.text,
            "stale": piece.stale,
        }

    @app.get("/repos/{repo_id}/chunks")
    def repo_chunks(repo_id: str, s: S, path: str = Query(min_length=1)) -> list[dict[str, Any]]:
        return [hit.to_dict() for hit in s.store.chunks_of(repo_id, path)]

    # ------------------------------------------------------------------ jobs
    @app.get("/jobs")
    def list_jobs(
        s: S, repo_id: str | None = None, limit: int = Query(50, ge=1, le=500)
    ) -> list[dict[str, Any]]:
        return [job.to_dict() for job in s.db.list_jobs(repo_id, limit)]

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, s: S) -> dict[str, Any]:
        job = s.db.get_job(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.to_dict()

    # ---------------------------------------------------------------- arama
    @app.post("/search")
    def search(body: SearchBody, s: S) -> dict[str, Any]:
        return s.retriever.search(body.to_request()).to_dict()

    @app.post("/ask")
    def ask(body: SearchBody, s: S) -> dict[str, Any]:
        if s.llm is None:
            raise HTTPException(
                503,
                "no LLM is configured — enter an Anthropic key under Connect › Keys, or "
                "install Ollama for a local model (RAG_LLM_PROVIDER=auto falls back to it)",
            )
        try:
            return run_ask(s.retriever, s.llm, body.to_request()).to_dict()
        except LLMError as error:
            raise HTTPException(502, str(error)) from error

    # -------------------------------------------------------------- webhook
    @app.post("/webhooks/azure/push")
    async def azure_push(
        request: Request,
        s: S,
        x_rag_webhook_secret: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
        secret: str | None = None,
    ) -> dict[str, Any]:
        if not verify_secret(s.webhook_secret, x_rag_webhook_secret, authorization, secret):
            raise HTTPException(401, "the webhook secret does not match")
        payload = await request.json()
        events = parse_azure_push(payload if isinstance(payload, dict) else {})
        if not events:
            return {
                "accepted": False,
                "reason": "not a git.push event, or no branch update",
            }
        return {"accepted": True, "results": _enqueue_push_events(s, events, "azure")}

    @app.post("/webhooks/github/push")
    async def github_push(
        request: Request,
        s: S,
        x_hub_signature_256: Annotated[str | None, Header()] = None,
        x_github_event: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        # The signature is computed over the raw body; the secret is the same RAG_WEBHOOK_SECRET.
        body = await request.body()
        if not verify_github_signature(s.webhook_secret, body, x_hub_signature_256):
            raise HTTPException(
                401, "X-Hub-Signature-256 did not verify (secret: RAG_WEBHOOK_SECRET)"
            )
        event_name = (x_github_event or "push").lower()
        if event_name == "ping":
            return {"accepted": True, "pong": True}
        if event_name != "push":
            return {"accepted": False, "reason": f"the {event_name} event is not handled"}
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            raise HTTPException(422, "invalid JSON") from None
        events = parse_github_push(payload if isinstance(payload, dict) else {})
        if not events:
            return {"accepted": False, "reason": "not a branch push (a tag or a delete)"}
        return {"accepted": True, "results": _enqueue_push_events(s, events, "github")}

    return app


def _llm_view(s: Services, key_source: str | None, probe: bool) -> dict[str, Any]:
    """Which LLM, from where, and is it working. With `probe`, a live ping: for Ollama the
    server + whether the model is pulled, for Anthropic whether the key is valid — the error
    text says what to do."""
    provider = s.llm.provider if s.llm else s.settings.resolved_llm_provider
    view: dict[str, Any] = {
        "mode": s.settings.llm_provider,  # auto | anthropic | openai | ollama
        "provider": provider,
        "model": s.llm.model if s.llm else s.settings.resolved_llm_model,
        "configured": s.llm is not None,
        "host": s.settings.ollama_host if provider == "ollama" else None,
        # Only the Anthropic key can be entered from the UI; OpenAI/Ollama come from the env.
        "source": key_source if provider == "anthropic" else None,
    }
    if s.llm is None:
        view["status"] = {"ok": False, "detail": "no LLM is configured"}
    elif not probe:
        view["status"] = {"ok": True, "detail": "not pinged (ask with probe=1)"}
    else:
        try:
            view["status"] = {"ok": True, "detail": s.llm.ping()}
        except LLMError as error:
            view["status"] = {"ok": False, "detail": str(error)}
    return view


def _enqueue_push_events(
    s: Services, events: list[PushEvent], provider: str
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in events:
        repo = s.db.find_repo_by_external(event.external_id, event.branch, provider=provider)
        if repo is None:
            results.append({"branch": event.branch, "ignored": "not a tracked repo/branch"})
            continue
        if event.new_commit and not s.db.record_webhook(repo.id, event.new_commit):
            results.append({"repo_id": repo.id, "ignored": "this commit was already handled"})
            continue
        job, created = s.jobs.enqueue(repo.id, "webhook")
        results.append({"repo_id": repo.id, "job_id": job.id, "new": created})
    return results


def _repo_view(s: Services, repo_id: str) -> dict[str, Any]:
    repo = s.db.get_repo(repo_id)
    assert repo is not None
    active = s.db.active_job(repo_id)
    return {**repo.to_dict(), "active_job": active.to_dict() if active else None}


app = create_app()
