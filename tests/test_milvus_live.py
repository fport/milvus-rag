"""The store layer against a real Milvus. No model needed: random vectors + BM25 text.

RAG_LIVE=1 uv run pytest -q tests/test_milvus_live.py
"""

import os
import uuid

import numpy as np
import pytest

from milvus_rag.index.store import MilvusStore, build_filter

pytestmark = pytest.mark.skipif(not os.getenv("RAG_LIVE"), reason="runs with RAG_LIVE=1")

DIM = 8


@pytest.fixture
def store():
    uri = os.getenv("RAG_MILVUS_URI", "http://localhost:19530")
    instance = MilvusStore(uri, f"test_{uuid.uuid4().hex[:8]}", DIM)
    if not instance.healthy():
        pytest.skip(f"Milvus yok: {uri}")
    instance.ensure_collection()
    yield instance
    instance.drop()


def _row(repo: str, path: str, ordinal: int, symbol: str, text: str, vector) -> dict:
    return {
        "id": f"{repo}:{path}#{ordinal}",
        "repo_id": repo,
        "path": path,
        "symbol": symbol,
        "parent_symbol": "",
        "kind": "function",
        "lang": "typescript",
        "category": "code",
        "ordinal": ordinal,
        "start_line": 1,
        "end_line": 5,
        "content": text,
        "indexed_text": f"// file: {path}\n// function: {symbol}\n{text}",
        "context": "",
        "symbols": symbol,
        "file_sha": "sha",
        "chunk_hash": "h",
        "indexed_at": "now",
        "dense": vector.tolist(),
    }


def test_insert_search_delete(store: MilvusStore):
    rng = np.random.default_rng(7)
    base = rng.normal(size=DIM).astype(np.float32)
    base /= np.linalg.norm(base)
    far = -base
    rows = [
        _row("r1", "src/auth.ts", 0, "withSession", "resolve session from request", base),
        _row("r1", "src/queue.ts", 0, "enqueueBoardSync", "add job to bullmq queue", far),
        _row("r2", "src/auth.ts", 0, "withSession", "another repo same path", far),
    ]
    assert store.insert(rows) == 3
    store.flush()
    assert store.count() == 3 and store.count("r1") == 2

    dense = store.dense_search(base, 3, build_filter(["r1"]))
    assert dense[0].path == "src/auth.ts" and dense[0].repo_id == "r1"
    assert dense[0].scores["dense"] > dense[1].scores["dense"]

    bm25 = store.bm25_search("bullmq queue", 3, build_filter(["r1"]))
    assert bm25 and bm25[0].symbol == "enqueueBoardSync" and "bm25" in bm25[0].scores

    by_symbol = store.bm25_search("withSession", 5)
    assert {hit.repo_id for hit in by_symbol} == {"r1", "r2"}

    only_r2 = store.bm25_search("withSession", 5, build_filter(["r2"]))
    assert [hit.repo_id for hit in only_r2] == ["r2"]

    assert [
        hit.ordinal if hasattr(hit, "ordinal") else 0
        for hit in store.chunks_of("r1", "src/auth.ts")
    ] == [0]

    store.delete_paths("r1", ["src/auth.ts"])
    assert store.count("r1") == 1 and store.count("r2") == 1
    store.delete_repo("r2")
    assert store.count() == 1
