"""The HTTP layer + the job queue, end to end with a fake embedder/store.

Without a real Milvus or model: registering a repo → the queue → the worker → the
manifest → the shape of /search → webhook verification and replay protection → the
limits on reading a file.
"""

import base64
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from milvus_rag.api import create_app
from milvus_rag.db import Database
from milvus_rag.index.pipeline import Indexer
from milvus_rag.jobs import JobRunner
from milvus_rag.models import Hit
from milvus_rag.repos import RepoService
from milvus_rag.search.retrieve import Retriever
from milvus_rag.services import Services


class FakeEmbedder:
    name = "fake"
    dimension = 4

    def encode(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    def encode_one(self, text):
        return np.ones(4, dtype=np.float32)

    def warm_up(self):
        pass


class FakeStore:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    def healthy(self):
        return True

    def ensure_collection(self):
        pass

    def insert(self, rows):
        for row in rows:
            self.rows[row["id"]] = row
        return len(rows)

    def delete_paths(self, repo_id, paths):
        for key in list(self.rows):
            if self.rows[key]["repo_id"] == repo_id and self.rows[key]["path"] in paths:
                del self.rows[key]

    def delete_repo(self, repo_id):
        for key in list(self.rows):
            if self.rows[key]["repo_id"] == repo_id:
                del self.rows[key]

    def count(self, repo_id=None):
        return sum(1 for r in self.rows.values() if repo_id is None or r["repo_id"] == repo_id)

    def _hits(self, channel, limit, expression):
        hits = []
        for row in list(self.rows.values())[:limit]:
            if expression and f'"{row["repo_id"]}"' not in expression:
                continue
            hits.append(
                Hit(
                    id=row["id"],
                    repo_id=row["repo_id"],
                    path=row["path"],
                    symbol=row["symbol"],
                    parent_symbol="",
                    kind=row["kind"],
                    lang=row["lang"],
                    category=row["category"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    content=row["content"],
                    context="",
                    scores={channel: 1.0},
                )
            )
        return hits

    def dense_search(self, vector, limit, expression=""):
        return self._hits("dense", limit, expression)

    def bm25_search(self, text, limit, expression=""):
        return self._hits("bm25", limit, expression)

    def chunks_of(self, repo_id, path):
        return []


@pytest.fixture
def client(tmp_settings, tmp_path: Path):
    db = Database(tmp_settings.db_path)
    store = FakeStore()
    embedder = FakeEmbedder()
    retriever = Retriever(tmp_settings, store, embedder, None)  # type: ignore[arg-type]
    indexer = Indexer(tmp_settings, db, store, embedder)  # type: ignore[arg-type]
    services = Services(
        settings=tmp_settings,
        db=db,
        store=store,  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        reranker=None,
        retriever=retriever,
        llm=None,
        enricher=None,
        azure=None,
        github=None,
        indexer=indexer,
        repos=RepoService(tmp_settings, db, store, None),  # type: ignore[arg-type]
        jobs=JobRunner(tmp_settings, db, indexer, retriever, None),
        webhook_secret=tmp_settings.webhook_secret,
    )
    repo_dir = tmp_path / "demo"
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "auth.ts").write_text(
        "export function withSession(c) {\n  return c;\n}\n" * 4
    )
    app = create_app(services=services, warm_up=False)
    with TestClient(app) as test_client:
        test_client.repo_dir = repo_dir  # type: ignore[attr-defined]
        yield test_client


def _wait_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "skipped"):
            return job
        time.sleep(0.05)
    raise AssertionError("the job never finished")


def test_register_index_search_and_sync(client: TestClient):
    assert client.get("/health").json()["status"] == "ok"

    created = client.post(
        "/repos", json={"provider": "local", "path": str(client.repo_dir), "name": "Demo Repo"}
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["id"] == "demo-repo" and body["job"]["status"] == "queued"

    job = _wait_job(client, body["job"]["id"])
    assert job["status"] == "done", job
    assert job["stats"]["added"] == 1 and job["stats"]["chunks_written"] >= 1

    repo = client.get("/repos/demo-repo").json()
    assert repo["status"] == "ready" and repo["file_count"] == 1 and repo["active_job"] is None
    assert client.get("/repos/demo-repo/files").json()[0]["path"] == "src/auth.ts"

    search = client.post("/search", json={"q": "withSession", "repo_ids": ["demo-repo"]}).json()
    assert search["mode"] == "bm25" and search["hits"][0]["symbol"] == "withSession"
    assert "bm25" in search["hits"][0]["scores"]
    prose = client.post("/search", json={"q": "how does auth work"}).json()
    assert prose["mode"] == "dense" and prose["hits"] and "dense" in prose["hits"][0]["scores"]
    fused = client.post("/search", json={"q": "how does auth work", "mode": "hybrid"}).json()
    assert fused["mode"] == "hybrid" and "rrf" in fused["hits"][0]["scores"]

    # Reading a file: the range, and escaping the directory.
    piece = client.get(
        "/repos/demo-repo/file", params={"path": "src/auth.ts", "start": 1, "end": 2}
    )
    assert piece.json()["text"].startswith("export function withSession")
    assert (
        client.get("/repos/demo-repo/file", params={"path": "../../etc/passwd"}).status_code == 404
    )

    # A second sync: nothing changed; a concurrent second request returns the same job.
    first = client.post("/repos/demo-repo/sync", json={"force": False})
    second = client.post("/repos/demo-repo/sync", json={"force": True})
    assert first.status_code == 202
    assert second.json()["job"]["id"] == first.json()["job"]["id"] or second.json()["created"]
    done = _wait_job(client, first.json()["job"]["id"])
    assert done["status"] == "done"

    assert client.post("/ask", json={"q": "x"}).status_code == 503
    assert client.delete("/repos/demo-repo").json() == {"deleted": "demo-repo"}
    assert client.get("/repos/demo-repo").status_code == 404


def test_webhook_auth_and_dedupe(client: TestClient):
    payload = {
        "eventType": "git.push",
        "resource": {
            "refUpdates": [{"name": "refs/heads/main", "oldObjectId": "a", "newObjectId": "b"}],
            "repository": {"id": "guid-1", "name": "Backend", "project": {"name": "P"}},
        },
    }
    assert client.post("/webhooks/azure/push", json=payload).status_code == 401

    basic = "Basic " + base64.b64encode(b"hook:s3cret").decode()
    unknown = client.post("/webhooks/azure/push", json=payload, headers={"Authorization": basic})
    assert unknown.status_code == 200 and unknown.json()["results"][0]["ignored"]

    # A registered repo opens a job; the same commit does not open a second one.
    services = client.app.state.services
    from milvus_rag.models import Repo

    services.db.upsert_repo(
        Repo(
            id="p-backend",
            name="Backend",
            provider="azure",
            branch="main",
            local_path="/tmp/none",
            external_id="guid-1",
            owner="P",
        )
    )
    first = client.post(
        "/webhooks/azure/push", json=payload, headers={"X-RAG-Webhook-Secret": "s3cret"}
    ).json()
    assert first["results"][0]["job_id"] and first["results"][0]["new"] is True
    again = client.post(
        "/webhooks/azure/push", json=payload, headers={"X-RAG-Webhook-Secret": "s3cret"}
    ).json()
    assert again["results"][0]["ignored"] == "this commit was already handled"


def test_github_webhook(client: TestClient):
    import json as jsonlib

    from milvus_rag.models import Repo
    from milvus_rag.webhooks import sign_github

    payload = {
        "ref": "refs/heads/main",
        "before": "a",
        "after": "b",
        "repository": {"id": 99, "name": "backend", "owner": {"login": "acme"}},
    }
    body = jsonlib.dumps(payload).encode()

    # Unsigned / wrongly signed → 401
    assert client.post("/webhooks/github/push", content=body).status_code == 401
    bad = client.post(
        "/webhooks/github/push",
        content=body,
        headers={"X-Hub-Signature-256": "sha256=deadbeef"},
    )
    assert bad.status_code == 401

    headers = {
        "X-Hub-Signature-256": sign_github("s3cret", body),
        "Content-Type": "application/json",
    }
    # ping → pong
    ping = client.post(
        "/webhooks/github/push",
        content=body,
        headers={**headers, "X-GitHub-Event": "ping"},
    )
    assert ping.json() == {"accepted": True, "pong": True}

    # An unregistered repo → ignored
    unknown = client.post("/webhooks/github/push", content=body, headers=headers)
    assert unknown.json()["results"][0]["ignored"]

    # A registered github repo → a job opens; the same commit does not open a second
    services = client.app.state.services
    services.db.upsert_repo(
        Repo(
            id="acme-backend",
            name="acme/backend",
            provider="github",
            branch="main",
            local_path="/tmp/none",
            external_id="99",
            owner="acme",
        )
    )
    first = client.post("/webhooks/github/push", content=body, headers=headers).json()
    assert first["results"][0]["job_id"] and first["results"][0]["new"] is True
    again = client.post("/webhooks/github/push", content=body, headers=headers).json()
    assert again["results"][0]["ignored"] == "this commit was already handled"


def test_home_serves_ui(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "Milvus RAG" in response.text
    assert "text/html" in response.headers["content-type"]
    # The tabs: the pipeline view and the integration guide are in the UI.
    assert "Index jobs" in response.text and "webhooks/github/push" in response.text

    health = client.get("/health").json()
    assert health["webhook_secret_set"] is True  # conftest sets the secret
    assert "poll_interval_seconds" in health


def test_credentials_flow(client: TestClient):
    # Initially: no credentials at all.
    state = client.get("/settings/credentials").json()
    assert state["github"]["token_set"] is False and state["azure"]["configured"] is False

    # Save the Azure credential (verification off: no network) → the client is built live.
    saved = client.put(
        "/settings/credentials",
        json={
            "azure_org_url": "https://dev.azure.com/acme",
            "azure_pat": "pat123",
            "verify": False,
        },
    )
    assert saved.status_code == 200
    state = client.get("/settings/credentials").json()
    assert state["azure"]["configured"] is True and state["azure"]["source"] == "ui"
    assert "acme" in state["azure"]["org_url"]

    # Save the GitHub token → /health reflects the live client.
    client.put("/settings/credentials", json={"github_token": "ghp_x", "verify": False})
    assert client.get("/health").json()["github_token"] is True

    # A field that is not sent is preserved; an empty string deletes it.
    client.put("/settings/credentials", json={"github_token": "", "verify": False})
    state = client.get("/settings/credentials").json()
    assert state["github"]["token_set"] is False
    assert state["azure"]["configured"] is True  # untouched

    # The webhook secret comes from the env ("s3cret"); a UI value overrides it, deleting reverts.
    assert state["webhook"] == {"secret_set": True, "source": "env"}
    push = {
        "eventType": "git.push",
        "resource": {"refUpdates": [], "repository": {"id": "x", "name": "x"}},
    }
    hook = "/webhooks/azure/push"
    assert (
        client.post(hook, json=push, headers={"X-RAG-Webhook-Secret": "s3cret"}).status_code == 200
    )
    client.put("/settings/credentials", json={"webhook_secret": "ui-secret", "verify": False})
    assert client.get("/settings/credentials").json()["webhook"]["source"] == "ui"
    assert (
        client.post(hook, json=push, headers={"X-RAG-Webhook-Secret": "s3cret"}).status_code == 401
    )
    assert (
        client.post(hook, json=push, headers={"X-RAG-Webhook-Secret": "ui-secret"}).status_code
        == 200
    )
    client.put("/settings/credentials", json={"webhook_secret": "", "verify": False})
    assert (
        client.post(hook, json=push, headers={"X-RAG-Webhook-Secret": "s3cret"}).status_code == 200
    )
    assert client.get("/health").json()["webhook_secret_set"] is True

    # The Anthropic key: a key entered from the UI builds the LLM live, and /health sees it.
    # (The starting state depends on the machine: the SDK may resolve a local profile too —
    # so only "did it come from the UI" is asserted.)
    # Provider auto → local Ollama with no key: the PUTs above already built the client.
    assert state["llm"]["mode"] == "auto" and state["llm"]["provider"] == "ollama"
    assert state["llm"]["configured"] is True and state["llm"]["host"] == "http://localhost:11434"
    assert state["llm"]["status"]["ok"] is True and "probe" in state["llm"]["status"]["detail"]
    client.put("/settings/credentials", json={"anthropic_api_key": "sk-ant-test", "verify": False})
    state = client.get("/settings/credentials").json()
    assert state["llm"]["provider"] == "anthropic" and state["llm"]["source"] == "ui"
    assert (
        state["llm"]["configured"] is True and "status" in state["llm"]
    )  # no probe: nothing is pinged
    assert client.get("/health").json()["llm"] == "claude-opus-5"
    client.put("/settings/credentials", json={"anthropic_api_key": "", "verify": False})
    state = client.get("/settings/credentials").json()
    assert state["llm"]["source"] != "ui" and state["llm"]["provider"] == "ollama"


MCP_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "1"},
    },
}
MCP_HEADERS = {
    "host": "localhost:8090",  # DNS-rebinding guard: only localhost is accepted
    "content-type": "application/json",
    "accept": "application/json, text/event-stream",
}


def test_mcp_endpoint(client: TestClient):
    # /mcp without a trailing slash must work without a redirect: some MCP clients lose
    # the body when they do not follow the 307.
    response = client.post("/mcp", json=MCP_INIT, headers=MCP_HEADERS, follow_redirects=False)
    assert response.status_code == 200, response.text
    assert '"name":"milvus-rag"' in response.text  # the initialize result streams as SSE
    assert client.post("/mcp/", json=MCP_INIT, headers=MCP_HEADERS).status_code == 200

    # A wrong accept header → the transport rejects it (not a 404: the endpoint is mounted).
    wrong = client.post(
        "/mcp", json=MCP_INIT, headers={**MCP_HEADERS, "accept": "application/json"}
    )
    assert wrong.status_code == 406

    # A foreign Host → the DNS-rebinding guard.
    foreign = client.post(
        "/mcp", json=MCP_INIT, headers={**MCP_HEADERS, "host": "evil.example.com"}
    )
    assert foreign.status_code == 421


def test_search_signal_and_file_guard(client: TestClient):
    """`/search` carries weak_match; `/repos/{id}/file` reads only an indexed file, says
    `stale` when it changed, and redacts secrets."""
    repo_dir: Path = client.repo_dir  # type: ignore[attr-defined]
    (repo_dir / "node_modules").mkdir()
    (repo_dir / "node_modules" / "x.js").write_text("module.exports = 1\n")
    (repo_dir / ".env").write_text("DB_PASSWORD=hunter2hunter2x1\n")
    created = client.post("/repos", json={"provider": "local", "path": str(repo_dir)})
    job = _wait_job(client, created.json()["job"]["id"])
    # .env and node_modules are outside the index.
    assert job["status"] == "done" and job["stats"]["added"] == 1

    search = client.post("/search", json={"q": "how does auth work"}).json()
    assert search["weak_match"] is False and search["dropped"] == 0  # sahte store dense=1.0

    fresh = client.get("/repos/demo/file", params={"path": "src/auth.ts", "end": 1}).json()
    assert fresh["stale"] is False and fresh["path"] == "src/auth.ts"
    for path in ("node_modules/x.js", ".env", "src/made-up.ts"):
        response = client.get("/repos/demo/file", params={"path": path})
        assert (
            response.status_code == 404
            and "is not indexed or does not exist" in response.json()["detail"]
        )

    (repo_dir / "src" / "auth.ts").write_text('const API_TOKEN="abc123def456ghi789";\n')
    edited = client.get("/repos/demo/file", params={"path": "src/auth.ts"}).json()
    assert edited["stale"] is True and "[SECRET]" in edited["text"]
