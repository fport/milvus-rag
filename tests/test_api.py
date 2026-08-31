"""HTTP katmanı + iş kuyruğu, sahte embedder/store ile uçtan uca.

Gerçek Milvus ve model olmadan: repo kaydı → kuyruk → worker → manifest →
/search şekli → webhook doğrulama ve tekrar koruması → dosya okuma sınırı.
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
    raise AssertionError("iş bitmedi")


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

    # Dosya okuma: aralık ve dizin dışına çıkma.
    piece = client.get(
        "/repos/demo-repo/file", params={"path": "src/auth.ts", "start": 1, "end": 2}
    )
    assert piece.json()["text"].startswith("export function withSession")
    assert (
        client.get("/repos/demo-repo/file", params={"path": "../../etc/passwd"}).status_code == 404
    )

    # İkinci sync: değişiklik yok; aynı anda ikinci istek aynı işi döndürür.
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

    # Repo kayıtlıysa iş açılır; aynı commit ikinci kez iş açmaz.
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
    assert again["results"][0]["ignored"] == "bu commit zaten işlendi"


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

    # İmzasız / yanlış imzalı → 401
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

    # Kayıtlı olmayan repo → ignored
    unknown = client.post("/webhooks/github/push", content=body, headers=headers)
    assert unknown.json()["results"][0]["ignored"]

    # Kayıtlı github reposu → iş açılır; aynı commit ikinci kez açmaz
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
    assert again["results"][0]["ignored"] == "bu commit zaten işlendi"


def test_home_serves_ui(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "Milvus RAG" in response.text
    assert "text/html" in response.headers["content-type"]
    # Sekmeler: pipeline görünümü ve entegrasyon rehberi arayüzde var.
    assert "Index işleri" in response.text and "webhooks/github/push" in response.text

    health = client.get("/health").json()
    assert health["webhook_secret_set"] is True  # conftest sırrı ayarlıyor
    assert "poll_interval_seconds" in health
