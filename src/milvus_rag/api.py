"""HTTP arayüzü.

    uv run rag serve            # ya da: uv run uvicorn milvus_rag.api:app --port 8090

Modeller lifespan'da bir kez yüklenir. Index işleri kuyruğa alınır ve arka
planda tek worker'da çalışır; /repos/{id} ve /jobs/{id} ilerlemeyi gösterir.
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
from milvus_rag.services import Services, apply_credentials, build_services
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

# MCP transport'unun DNS rebinding koruması: varsayılan localhost kümesi.
_LOCAL_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOCAL_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


# ------------------------------------------------------------------- şemalar


class AddRepoBody(BaseModel):
    provider: Literal["azure", "github", "local", "git"] = "azure"
    # azure / github
    project: str | None = None
    repo: str | None = Field(
        default=None, description="Azure: repo adı ya da GUID · GitHub: owner/repo"
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
    """Gönderilmeyen alana dokunulmaz; boş string kaydı siler (env'e dönülür)."""

    github_token: str | None = None
    azure_org_url: str | None = None
    azure_pat: str | None = None
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
    # MCP sunucusu uygulamayla aynı süreçte, aynı retriever'ın üstünde çalışır.
    # Servisler lifespan'da doğduğu için araçlar onları tembel çözer.
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
            log.warning("Milvus'a ulaşılamıyor; /search boş dönecek", uri=built.settings.milvus_uri)
        await built.jobs.start()
        log.info(
            "hazır",
            port=built.settings.port,
            collection=built.settings.milvus_collection,
            embedding=built.embedder.name,
            rerank=built.reranker.model_name if built.reranker else "kapalı",
            azure="açık" if built.azure else "kapalı",
            llm=built.llm.model if built.llm else "kapalı",
            mcp="/mcp",
        )
        # Oturum yöneticisi mount edilen alt uygulamanın kendi lifespan'ıyla değil,
        # buradan çalışır (FastAPI mount'ta alt lifespan'ı çağırmaz).
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

    # ---------------------------------------------------------------- arayüz
    static_dir = Path(__file__).parent / "static"

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    # ---------------------------------------------------------------- sağlık
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
            "webhook_secret_set": bool(s.settings.webhook_secret),
            "poll_interval_seconds": s.settings.poll_interval_seconds,
        }

    # ------------------------------------------------------------- kimlikler
    @app.get("/settings/credentials")
    def get_credentials(s: S) -> dict[str, Any]:
        """Maskelenmiş durum: hangi kimlik nereden geliyor. Sırların kendisi dönmez."""
        saved = s.db.get_app_settings(["github_token", "azure_org_url", "azure_pat"])

        def source(key: str, env_value: str | None) -> str | None:
            if saved.get(key):
                return "ui"
            return "env" if env_value else None

        return {
            "github": {
                "token_set": bool(s.github and s.github.token),
                "source": source("github_token", s.settings.github_token),
            },
            "azure": {
                "configured": s.azure is not None,
                "org_url": s.azure.org_url if s.azure else (s.settings.azure_org_url or ""),
                "source": source("azure_pat", s.settings.azure_pat),
            },
        }

    @app.put("/settings/credentials")
    def put_credentials(body: CredentialsBody, s: S) -> dict[str, Any]:
        """Kimlikleri kaydet, istemcileri canlı yeniden kur, istenirse doğrula.

        Sırlar data/rag.db'de düz metin durur — servis zaten iç ağ içindir
        (bkz. DEPLOYMENT.md). Boş string gönderilen alan silinir.
        """
        for key in ("github_token", "azure_org_url", "azure_pat"):
            if key in body.model_fields_set:
                value = getattr(body, key)
                s.db.set_app_setting(key, (value or "").strip() or None)
        apply_credentials(s)

        result: dict[str, Any] = {"github": None, "azure": None}
        if body.verify:
            if s.github and s.github.token:
                try:
                    result["github"] = {"ok": True, "login": s.github.whoami()}
                except GitHubError as error:
                    result["github"] = {"ok": False, "detail": str(error)}
            if s.azure is not None:
                try:
                    projects = s.azure.list_projects()
                    result["azure"] = {"ok": True, "projects": len(projects)}
                except AzureError as error:
                    result["azure"] = {"ok": False, "detail": str(error)}
        return result

    # ----------------------------------------------------------------- azure
    @app.get("/azure/projects")
    def azure_projects(s: S) -> list[dict[str, Any]]:
        if s.azure is None:
            raise HTTPException(503, "Azure DevOps yapılandırılmamış")
        return [project.to_dict() for project in s.azure.list_projects()]

    @app.get("/azure/projects/{project}/repos")
    def azure_repos(project: str, s: S) -> list[dict[str, Any]]:
        if s.azure is None:
            raise HTTPException(503, "Azure DevOps yapılandırılmamış")
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
            raise HTTPException(503, "GitHub istemcisi kurulmamış")
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
                raise HTTPException(422, "azure için project ve repo gerekli")
            repo = s.repos.register_azure(body.project, body.repo, body.branch, body.auto_sync)
        elif body.provider == "github":
            if not body.repo:
                raise HTTPException(422, "github için repo ('owner/repo') gerekli")
            repo = s.repos.register_github(body.repo, body.branch, body.auto_sync)
        elif body.provider == "local":
            if not body.path:
                raise HTTPException(422, "local için path gerekli")
            repo = s.repos.register_local(body.path, body.name, body.auto_sync)
        else:
            if not body.url:
                raise HTTPException(422, "git için url gerekli")
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
        """Bir dosyanın satır aralığı — agent'ın `read_code` aracı için.

        Yalnızca indexlenmiş dosya okunur (manifest); `stale` diskteki içeriğin
        son indexten sonra değiştiğini söyler. Gerekçe: sources/files.py.
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
            raise HTTPException(404, "iş yok")
        return job.to_dict()

    # ---------------------------------------------------------------- arama
    @app.post("/search")
    def search(body: SearchBody, s: S) -> dict[str, Any]:
        return s.retriever.search(body.to_request()).to_dict()

    @app.post("/ask")
    def ask(body: SearchBody, s: S) -> dict[str, Any]:
        if s.llm is None:
            raise HTTPException(503, "LLM yapılandırılmamış (RAG_LLM_PROVIDER / API anahtarı)")
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
        if not verify_secret(
            s.settings.webhook_secret, x_rag_webhook_secret, authorization, secret
        ):
            raise HTTPException(401, "webhook sırrı eşleşmiyor")
        payload = await request.json()
        events = parse_azure_push(payload if isinstance(payload, dict) else {})
        if not events:
            return {
                "accepted": False,
                "reason": "git.push olayı değil ya da branch güncellemesi yok",
            }
        return {"accepted": True, "results": _enqueue_push_events(s, events, "azure")}

    @app.post("/webhooks/github/push")
    async def github_push(
        request: Request,
        s: S,
        x_hub_signature_256: Annotated[str | None, Header()] = None,
        x_github_event: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        # İmza ham gövde üstünden hesaplanır; sır RAG_WEBHOOK_SECRET ile aynı.
        body = await request.body()
        if not verify_github_signature(s.settings.webhook_secret, body, x_hub_signature_256):
            raise HTTPException(401, "X-Hub-Signature-256 doğrulanamadı (sır: RAG_WEBHOOK_SECRET)")
        event_name = (x_github_event or "push").lower()
        if event_name == "ping":
            return {"accepted": True, "pong": True}
        if event_name != "push":
            return {"accepted": False, "reason": f"{event_name} olayı işlenmez"}
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            raise HTTPException(422, "geçersiz JSON") from None
        events = parse_github_push(payload if isinstance(payload, dict) else {})
        if not events:
            return {"accepted": False, "reason": "branch push'u değil (tag ya da silme)"}
        return {"accepted": True, "results": _enqueue_push_events(s, events, "github")}

    return app


def _enqueue_push_events(
    s: Services, events: list[PushEvent], provider: str
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in events:
        repo = s.db.find_repo_by_external(event.external_id, event.branch, provider=provider)
        if repo is None:
            results.append({"branch": event.branch, "ignored": "izlenen repo/branch değil"})
            continue
        if event.new_commit and not s.db.record_webhook(repo.id, event.new_commit):
            results.append({"repo_id": repo.id, "ignored": "bu commit zaten işlendi"})
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
