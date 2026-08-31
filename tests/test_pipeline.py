"""Artımlı index akışı, sahte embedder ve sahte Milvus ile.

Milvus'a gerek yok: depo bir dict. Ölçülen şey akışın kendisi — değişen dosya
yeniden yazılıyor mu, silinen siliniyor mu, aynı kalan embed ediliyor mu.
"""

from pathlib import Path

import numpy as np

from milvus_rag.index.pipeline import Indexer
from milvus_rag.models import Repo


class FakeEmbedder:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[int] = []

    @property
    def dimension(self) -> int:
        return 4

    def encode(self, texts):
        self.calls.append(len(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    def encode_one(self, text):
        return self.encode([text])[0]

    def warm_up(self) -> None:
        pass


class FakeStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.deleted: list[tuple[str, tuple[str, ...]]] = []

    def ensure_collection(self) -> None:
        pass

    def insert(self, rows):
        for row in rows:
            self.rows[row["id"]] = row
        return len(rows)

    def delete_paths(self, repo_id, paths):
        self.deleted.append((repo_id, tuple(paths)))
        for key in list(self.rows):
            if self.rows[key]["repo_id"] == repo_id and self.rows[key]["path"] in paths:
                del self.rows[key]

    def delete_repo(self, repo_id):
        for key in list(self.rows):
            if self.rows[key]["repo_id"] == repo_id:
                del self.rows[key]

    def count(self, repo_id=None):
        return sum(1 for row in self.rows.values() if repo_id is None or row["repo_id"] == repo_id)


def _write(root: Path, name: str, text: str) -> None:
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text)


def test_incremental_sync(tmp_settings, db, tmp_path: Path):
    root = tmp_path / "repo"
    _write(root, "src/a.ts", "export function a() { return 1 }\n" * 3)
    _write(root, "src/b.ts", "export function b() { return 2 }\n" * 3)
    _write(root, "README.md", "# Demo\n\nhello\n")
    _write(root, "node_modules/x.js", "ignored")

    repo = db.upsert_repo(
        Repo(id="demo", name="demo", provider="local", branch="", local_path=str(root))
    )
    embedder = FakeEmbedder()
    store = FakeStore()
    indexer = Indexer(tmp_settings, db, store, embedder)  # type: ignore[arg-type]

    first = indexer.sync(repo)
    assert first.files_seen == 3 and first.added == 3 and first.deleted == 0
    assert first.chunks_written == store.count("demo") > 0
    assert set(db.manifest("demo")) == {"src/a.ts", "src/b.ts", "README.md"}
    assert db.get_repo("demo").status == "ready"
    assert db.get_repo("demo").index_version == tmp_settings.index_version

    # Değişiklik yok → hiçbir şey embed edilmez.
    embedder.calls.clear()
    second = indexer.sync(db.get_repo("demo"))
    assert second.unchanged == 3 and second.added == second.modified == second.deleted == 0
    assert embedder.calls == []

    # Bir dosya değişti, biri silindi, biri eklendi.
    _write(root, "src/a.ts", "export function a() { return 42 }\n" * 3)
    (root / "src/b.ts").unlink()
    _write(root, "src/c.ts", "export const c = 3\n")
    third = indexer.sync(db.get_repo("demo"))
    assert (third.modified, third.deleted, third.added, third.unchanged) == (1, 1, 1, 1)
    paths = {row["path"] for row in store.rows.values()}
    assert paths == {"src/a.ts", "src/c.ts", "README.md"}
    assert any("42" in row["content"] for row in store.rows.values())
    assert ("demo", ("src/b.ts",)) in store.deleted
    assert set(db.manifest("demo")) == paths

    # Index sürümü değişti → zorla tam yeniden index.
    db.update_repo("demo", index_version="eski")
    forced = indexer.sync(db.get_repo("demo"))
    assert forced.forced and forced.reason and forced.added == 3


def test_rows_carry_symbol_and_indexed_text(tmp_settings, db, tmp_path: Path):
    root = tmp_path / "repo"
    _write(root, "auth.ts", "/** doc */\nexport async function withSession(c) {\n  return c\n}\n")
    repo = db.upsert_repo(Repo(id="r", name="r", provider="local", branch="", local_path=str(root)))
    store = FakeStore()
    Indexer(tmp_settings, db, store, FakeEmbedder()).sync(repo)  # type: ignore[arg-type]
    row = next(iter(store.rows.values()))
    assert row["symbol"] == "withSession" and row["lang"] == "typescript"
    assert row["indexed_text"].startswith("// file: auth.ts")
    assert row["id"] == "r:auth.ts#0" and len(row["dense"]) == 4
