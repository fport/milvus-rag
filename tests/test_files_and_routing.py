from pathlib import Path

from milvus_rag.index.scrub import scrub
from milvus_rag.models import Hit
from milvus_rag.search.retrieve import fuse_rrf
from milvus_rag.search.routing import looks_like_symbol
from milvus_rag.sources.files import (
    category_for,
    is_indexable_path,
    iter_source_files,
    language_for,
    parse_extra_extensions,
)


def test_indexable_paths():
    assert is_indexable_path("src/a.ts")
    assert is_indexable_path("Dockerfile")
    assert is_indexable_path("docs/README.md")
    assert not is_indexable_path("node_modules/x/index.js")
    assert not is_indexable_path("dist/app.min.js")
    assert not is_indexable_path("src/types.d.ts")
    assert not is_indexable_path("package-lock.json")
    assert not is_indexable_path("image.png")
    assert not is_indexable_path("Foo.Designer.cs")
    assert is_indexable_path("schema.proto", parse_extra_extensions("proto, graphql"))


def test_language_and_category():
    assert language_for("a/b.tsx") == "tsx"
    assert language_for("Program.cs") == "csharp"
    assert language_for("x.unknown") is None
    assert category_for("README.mdx") == "doc"
    assert category_for("main.go") == "code"
    assert category_for("config.yaml") == "other"
    assert language_for("m.sql") is None and category_for("m.sql") == "code"


def test_iter_source_files_skips_binary_and_large(tmp_path: Path):
    (tmp_path / "ok.py").write_text("print(1)\n")
    (tmp_path / "bin.py").write_bytes(b"\x00\x01binary")
    (tmp_path / "big.py").write_text("x" * 10_000)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("module.exports = 1")
    found = {source.path: source for source in iter_source_files(tmp_path, max_bytes=5_000)}
    assert set(found) == {"ok.py"}
    assert found["ok.py"].sha and found["ok.py"].lang == "python"


def test_symbol_routing():
    assert looks_like_symbol("handleAuthCallback")
    assert looks_like_symbol("QUEUE_NAMES")
    assert looks_like_symbol("auth.service")
    assert looks_like_symbol("jira-board-sync")
    assert looks_like_symbol("src/middlewares/auth.ts")
    assert not looks_like_symbol("auth")
    assert not looks_like_symbol("how does auth work")
    assert not looks_like_symbol("Kimlik doğrulama nasıl çalışıyor?")


def _hit(id_: str, **scores: float) -> Hit:
    return Hit(
        id=id_,
        repo_id="r",
        path=id_,
        symbol="",
        parent_symbol="",
        kind="",
        lang="",
        category="",
        start_line=1,
        end_line=2,
        content="",
        context="",
        scores=scores,
    )


def test_rrf_keeps_channel_scores_and_prefers_agreement():
    dense = [_hit("a", dense=0.9), _hit("b", dense=0.8), _hit("c", dense=0.7)]
    bm25 = [_hit("c", bm25=12.0), _hit("d", bm25=9.0), _hit("a", bm25=3.0)]
    fused = fuse_rrf(dense, bm25, k=60)
    assert [hit.id for hit in fused][:2] == ["a", "c"]  # iki kanalda da var
    top = fused[0]
    assert top.scores["dense"] == 0.9 and top.scores["bm25"] == 3.0 and "rrf" in top.scores
    assert fused[-1].rank == len(fused) - 1


def test_scrub_redacts_real_secrets_only():
    text = (
        "DB_PASSWORD=aB3dE5fG7hI9jK1lM\n"
        "token: text('token')\n"
        "const key = process.env.OPENAI_API_KEY\n"
        "mail: someone@company.com and admin@example.com\n"
        "sk-abcdefghijklmnopqrstuvwxyz123456\n"
    )
    result = scrub(text)
    assert "DB_PASSWORD=[SECRET]" in result.text
    assert "token: text('token')" in result.text
    assert "process.env.OPENAI_API_KEY" in result.text
    assert "someone@company.com" not in result.text and "admin@example.com" in result.text
    assert "[API_KEY]" in result.text
    assert result.total == 3
