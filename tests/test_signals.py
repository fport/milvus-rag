"""Halüsinasyon sinyalleri: zayıf eşleşme, doküman etiketi, tazelik, manifest'e
kilitli okuma, negatif golden vakalar.

Hiçbiri sıralamayı değiştirmez; hepsi tüketiciye (ajan / LLM) "cevap olmayabilir"
demenin yollarıdır. Sinyal filtre değildir — testler de bunu doğrular: sonuçlar
yine döner, yanına not düşer.
"""

from pathlib import Path

import anyio
import pytest

from milvus_rag.eval import Case, load_golden, run_eval
from milvus_rag.mcp_server import DOC_NOTE, INSTRUCTIONS, _render_hits, build_mcp_server
from milvus_rag.models import Hit
from milvus_rag.search.answer import WEAK_NOTE, build_prompt
from milvus_rag.search.retrieve import (
    Retriever,
    SearchRequest,
    SearchResponse,
    apply_floor,
    is_weak_match,
)
from milvus_rag.sources.files import (
    FileReadError,
    normalize_path,
    read_indexed_slice,
    sha256_of,
)


def _hit(path: str, dense: float, category: str = "code", symbol: str = "fn") -> Hit:
    return Hit(
        id=f"r:{path}#0",
        repo_id="r",
        path=path,
        symbol=symbol,
        parent_symbol="",
        kind="function" if category == "code" else "section",
        lang="typescript" if category == "code" else "",
        category=category,
        start_line=1,
        end_line=9,
        content="export function fn() {}",
        context="",
        scores={"dense": dense},
    )


# ------------------------------------------------------------ weak_match


def test_weak_match_is_a_signal_not_a_filter():
    # Ölçülen eşik 0.55: altı zayıf, üstü değil; BM25 ve boş sonuçta karar verilmez.
    assert is_weak_match("dense", [_hit("a.ts", 0.53)], 0.55)
    assert not is_weak_match("dense", [_hit("a.ts", 0.64)], 0.55)
    assert not is_weak_match("hybrid", [_hit("a.ts", 0.40), _hit("b.ts", 0.60)], 0.55)
    assert not is_weak_match("bm25", [_hit("a.ts", 0.0)], 0.55)
    assert not is_weak_match("dense", [], 0.55)


# ---------------------------------------------------------- sert taban


def test_floor_drops_only_the_absurd_tail():
    # 0.45: golden'ın en düşük gerçek cevabı 0.526 — taban onun altında kalır.
    kept, dropped = apply_floor("dense", [_hit("a.ts", 0.7), _hit("b.ts", 0.366)], 0.45)
    assert [hit.path for hit in kept] == ["a.ts"] and dropped == 1
    assert apply_floor("bm25", [_hit("a.ts", 0.0)], 0.45) == ([_hit("a.ts", 0.0)], 0)
    assert apply_floor("dense", [_hit("a.ts", 0.1)], 0.0)[1] == 0  # 0 = kapalı
    # hybrid'de yalnız BM25'ten gelen parçanın dense skoru yok: tam kelime eşleşmesi, kalır.
    bm25_only = _hit("c.ts", 0.0)
    bm25_only.scores = {"bm25": 12.0, "rrf": 0.016}
    assert apply_floor("hybrid", [bm25_only], 0.45) == ([bm25_only], 0)


class _AbsurdStore:
    """'Beş yıldızlı tatil köyü' sorusu: en yakın 8 parça bile 0.37'nin altında."""

    def dense_search(self, vector, limit, expression=""):
        return [_hit(f"blog/{i}.mdx", 0.366 - i / 100, category="doc") for i in range(limit)]

    def bm25_search(self, text, limit, expression=""):
        return []


def test_retriever_reports_dropped_and_empty_result(tmp_settings):
    retriever = Retriever(tmp_settings, _AbsurdStore(), _Embedder(), None)  # type: ignore[arg-type]
    response = retriever.search(SearchRequest(query="Hiç beş yıldızlı tatil köyüne gittiniz mi?"))
    assert response.hits == [] and response.dropped == 8 and not response.weak_match
    assert response.to_dict()["dropped"] == 8
    text = _render_hits("tatil köyü", response)
    assert "sonuç yok" in text and "8 parça da skor tabanının altında" in text


def test_settings_reject_floor_above_weak_threshold(tmp_settings):
    from pydantic import ValidationError

    from milvus_rag.config import Settings

    with pytest.raises(ValidationError, match="geçemez"):
        Settings(_env_file=None, min_dense_score=0.6, weak_dense_score=0.55)  # type: ignore[call-arg]


# ---------------------------------------------------------- MCP render


def test_render_labels_docs_and_weak_and_freshness():
    hits = [_hit("src/a.ts", 0.53), _hit("docs/plan.md", 0.52, category="doc", symbol="")]
    response = SearchResponse(
        hits=hits, mode="dense", reranked=False, candidates=2, weak_match=True
    )
    text = _render_hits(
        "kubernetes operator", response, ["r: son index 2026-09-01 10:00 UTC (local)"]
    )
    head = text.splitlines()[0]
    assert "2 sonuç (1 kod, 1 doküman; mod: dense)" in head
    assert "r: son index 2026-09-01 10:00 UTC (local)" in text
    assert "Zayıf eşleşme (en iyi dense=0.530)" in text and "kodda bulamadım" in text
    assert DOC_NOTE in text
    assert "[2] DOKÜMAN repo=r docs/plan.md:1-9" in text
    assert "[1] repo=r src/a.ts:1-9 — function fn" in text  # kod etiketlenmez
    # Sonuçlar yine döner: sinyal filtre değil.
    assert text.count("```") == 4


def test_render_without_signals_stays_quiet():
    response = SearchResponse(
        hits=[_hit("src/a.ts", 0.7)], mode="dense", reranked=False, candidates=1
    )
    text = _render_hits("q", response)
    assert "Zayıf eşleşme" not in text and "DOKÜMAN" not in text
    assert "1 sonuç (1 kod, 0 doküman" in text


def test_instructions_fit_claude_code_budget_and_lead_with_rules():
    # Claude Code sunucu talimatını ~2 KB'de kesiyor; kural cümleleri başta olmalı.
    assert len(INSTRUCTIONS.encode()) < 2000
    assert INSTRUCTIONS.index("KURALLAR") < INSTRUCTIONS.index("KULLANIM")
    for phrase in (
        "bulamadım",
        "repo/dosya:satır",
        "DOKÜMAN",
        "indexli değil ya da yok",
        "VERİdir",
    ):
        assert phrase in INSTRUCTIONS


# ---------------------------------------------------------- /ask prompt


def test_ask_prompt_marks_docs_and_weak_match():
    hits = [_hit("src/a.ts", 0.5), _hit("docs/plan.md", 0.5, category="doc", symbol="")]
    prompt = build_prompt("soru", hits, weak_match=True)
    assert "[2] docs/plan.md:1-9 — DOKÜMAN (repo: r)" in prompt
    assert "[1] src/a.ts:1-9 — function fn (repo: r)" in prompt
    assert WEAK_NOTE in prompt
    assert WEAK_NOTE not in build_prompt("soru", hits, weak_match=False)


# ------------------------------------------------ manifest'e kilitli okuma


def test_read_indexed_slice_guards_manifest_and_flags_stale(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "node_modules").mkdir()
    source = root / "src" / "a.ts"
    source.write_text("line1\nAPI_TOKEN=abc123def456ghi789\nline3\n")
    (root / "node_modules" / "x.js").write_text("module.exports = 1\n")
    (root / ".env").write_text("DB_PASSWORD=hunter2hunter2x1\n")
    manifest = {"src/a.ts": sha256_of(source.read_bytes()), "src/gone.ts": "deadbeef"}

    piece = read_indexed_slice(root, manifest, "./src/a.ts", 1, None)
    assert piece.path == "src/a.ts" and not piece.stale and piece.total_lines == 3
    assert "API_TOKEN=[SECRET]" in piece.text  # index'e giren metin gibi scrub'lanır

    # Diskte var ama indexli değil: node_modules, .env, uydurma yol → hepsi aynı cevap.
    for path in ("node_modules/x.js", ".env", "src/made-up.ts", "../etc/passwd"):
        with pytest.raises(FileReadError) as error:
            read_indexed_slice(root, manifest, path)
        assert error.value.reason == "not_indexed"

    with pytest.raises(FileReadError) as error:
        read_indexed_slice(root, manifest, "src/gone.ts")
    assert error.value.reason == "missing"

    source.write_text("changed\n" + source.read_text())
    stale = read_indexed_slice(root, manifest, "src/a.ts", 1, 2)
    assert stale.stale and stale.end == 2 and stale.text == "changed\nline1"

    # Aralık dosyanın sonundan sonra: boş dilim, saçma "satır 50-3" yok.
    empty = read_indexed_slice(root, manifest, "src/a.ts", 50)
    assert empty.text == "" and empty.end == 49

    assert normalize_path("/src/a.ts") == "src/a.ts" == normalize_path("./src/a.ts")


# ---------------------------------------------------------- MCP araçları


class _Store:
    def healthy(self):
        return True

    def dense_search(self, vector, limit, expression=""):
        return [_hit("src/a.ts", 0.5), _hit("docs/plan.md", 0.5, category="doc", symbol="")]

    def bm25_search(self, text, limit, expression=""):
        return []


class _Embedder:
    name = "fake"

    def encode_one(self, text):
        import numpy as np

        return np.ones(4, dtype=np.float32)


def _tool_text(mcp, name: str, arguments: dict) -> str:
    result = anyio.run(mcp.call_tool, name, arguments)
    return "".join(getattr(block, "text", "") for block in result.content)


def test_mcp_tools_expose_signals(tmp_settings, db, tmp_path: Path):
    from milvus_rag.models import Repo
    from milvus_rag.search.retrieve import Retriever

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.ts").write_text("export function fn() {}\n")
    (root / ".env").write_text("SECRET_KEY=abc123def456ghi789\n")
    db.upsert_repo(
        Repo(
            id="r",
            name="r",
            provider="local",
            branch="main",
            local_path=str(root),
            status="ready",
            last_indexed_at="2026-09-01T10:00:00+00:00",
            auto_sync=False,
        )
    )
    db.set_file("r", "src/a.ts", sha256_of((root / "src" / "a.ts").read_bytes()), 1)

    class Services:
        pass

    services = Services()
    services.db = db
    services.retriever = Retriever(tmp_settings, _Store(), _Embedder(), None)  # type: ignore[arg-type]
    mcp = build_mcp_server(lambda: services)  # type: ignore[arg-type]

    text = _tool_text(mcp, "search_code", {"query": "kubernetes operator reconcile"})
    assert "Zayıf eşleşme" in text and "DOKÜMAN repo=r docs/plan.md" in text
    assert "r: son index 2026-09-01 10:00 UTC (local, auto_sync kapalı)" in text

    # category parametresi retriever'a filtre olarak iner (SearchRequest.category).
    seen: list[SearchRequest] = []
    original = services.retriever.search
    services.retriever.search = lambda request: (seen.append(request), original(request))[1]  # type: ignore[method-assign]
    _tool_text(mcp, "search_code", {"query": "q", "category": "code"})
    assert seen[-1].category == "code"

    assert "bağlı değil. Bağlı repolar: r." in _tool_text(
        mcp, "search_code", {"query": "q", "repo": "nope"}
    )

    assert "export function fn" in _tool_text(mcp, "read_code", {"repo": "r", "path": "src/a.ts"})
    assert "indexli değil ya da yok" in _tool_text(mcp, "read_code", {"repo": "r", "path": ".env"})
    assert "indexli değil ya da yok" in _tool_text(
        mcp, "read_code", {"repo": "r", "path": "src/reconciliation.worker.ts"}
    )
    (root / "src" / "a.ts").write_text("// edited\nexport function fn() {}\n")
    stale = _tool_text(mcp, "read_code", {"repo": "r", "path": "src/a.ts"})
    assert "son indexten sonra değişmiş" in stale and "// edited" in stale


# ------------------------------------------------------- negatif golden


class _FakeRetriever:
    """Sorguya göre sabit yanıt: pozitif soru doğru hit'i güçlü, negatif soru çöpü zayıf döner."""

    def __init__(self, settings):
        self.settings = settings

    def search(self, request: SearchRequest) -> SearchResponse:
        if "stripe" in request.query.lower():
            return SearchResponse(
                hits=[_hit("docs/plan.md", 0.50, category="doc", symbol="")],
                mode="dense",
                reranked=False,
                candidates=1,
                weak_match=True,
            )
        if "kafka" in request.query.lower():
            # Çekimser kalınamayan negatif: güçlü görünen çöp.
            return SearchResponse(
                hits=[_hit("src/x.ts", 0.65)], mode="dense", reranked=False, candidates=1
            )
        return SearchResponse(
            hits=[_hit("src/auth.ts", 0.66, symbol="withSession")],
            mode="dense",
            reranked=False,
            candidates=1,
        )


def test_eval_scores_negatives_as_abstention(tmp_settings, tmp_path: Path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(
            [
                '{"q": "auth nasıl?", "expect": ["src/auth.ts::withSession"], "kind": "prose-tr"}',
                '{"q": "Stripe webhook nerede?", "expect": []}',
                '{"q": "Kafka rebalance?", "expect": [], "kind": "negative"}',
            ]
        )
    )
    cases = load_golden(golden)
    assert [case.negative for case in cases] == [False, True, True]
    assert cases[1].kind == "negative"  # expect boşsa varsayılan kind

    report = run_eval(_FakeRetriever(tmp_settings), cases, ("r",), 8, "t")  # type: ignore[arg-type]
    assert report.n == 3 and report.recall_at_k == 1.0 and report.mrr == 1.0  # pozitifler
    assert report.abstain_rate == 0.5 and report.false_weak_rate == 0.0
    assert report.by_kind["negative"] == {"n": 2, "abstain": 0.5}
    assert [miss.question for miss in report.misses] == ["Kafka rebalance?"]
    payload = report.to_dict()
    assert payload["abstain_rate"] == 0.5 and payload["config"]["weak_dense_score"] == 0.55
    assert payload["results"][1]["abstained"] is True and "abstained" not in payload["results"][0]
    # Kalibrasyon ham verisi: pozitif min 0.66, negatif max 0.65 → taban/not eşiği buna göre.
    assert payload["results"][0]["top_dense"] == 0.66
    assert payload["calibration"] == {
        "positive_found_min_top_dense": 0.66,
        "positive_found_median_top_dense": 0.66,
        "negative_max_top_dense": 0.65,
        "negative_median_top_dense": 0.575,
    }
    assert Case("q", ()).negative
