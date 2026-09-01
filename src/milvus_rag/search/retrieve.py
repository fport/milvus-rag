"""Retrieval zinciri: yönlendir → kanal(lar) → RRF → rerank → top-k.

Her parça bir ayarla kapanabilir (ablation için). Her sonuç kanal bazında
skorlarını taşır: `dense`, `bm25`, `rrf`, `rerank`. Hangi kanalın neyi bulduğu
veri yapısında görünür; eval ve hata ayıklama buna dayanır.

RRF iki listenin skorlarını atar, sıralarını kullanır: Σ 1/(k + rank). Cosine
(0-1) ve BM25 (0-30) farklı ölçekte olduğundan toplanamaz.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from milvus_rag.config import Settings
from milvus_rag.index.embed import Embedder
from milvus_rag.index.store import MilvusStore, build_filter
from milvus_rag.log import get_logger
from milvus_rag.models import Hit
from milvus_rag.search.rerank import CrossEncoderReranker
from milvus_rag.search.routing import looks_like_symbol

log = get_logger("retrieve")

Mode = Literal["auto", "dense", "bm25", "hybrid"]


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    repo_ids: tuple[str, ...] = ()
    k: int | None = None
    mode: Mode | None = None
    rerank: bool | None = None
    candidates: int | None = None
    path_prefix: str = ""
    lang: str = ""
    category: str = ""

    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.query.strip(),
            self.repo_ids,
            self.k,
            self.mode,
            self.rerank,
            self.candidates,
            self.path_prefix,
            self.lang,
            self.category,
        )


@dataclass(slots=True)
class SearchResponse:
    hits: list[Hit]
    mode: str
    reranked: bool
    candidates: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    cached: bool = False
    # En iyi dense skoru `weak_dense_score`'un altında: sonuçlar döner ama tüketici
    # "bulamadım" demeyi düşünmeli. Filtre değil sinyal — bkz. config.
    weak_match: bool = False
    # `min_dense_score` tabanının altında kaldığı için atılan parça sayısı: "8 aday
    # vardı, hepsi saçmaydı" ile "hiç aday yoktu" tüketici için farklı cümleler.
    dropped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": [hit.to_dict() for hit in self.hits],
            "mode": self.mode,
            "reranked": self.reranked,
            "candidates": self.candidates,
            "timings_ms": {key: round(value, 1) for key, value in self.timings_ms.items()},
            "cached": self.cached,
            "weak_match": self.weak_match,
            "dropped": self.dropped,
        }


class _TTLCache:
    def __init__(self, size: int, ttl: float) -> None:
        self.size = size
        self.ttl = ttl
        self._items: OrderedDict[tuple[Any, ...], tuple[float, SearchResponse]] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> SearchResponse | None:
        if self.size == 0 or self.ttl == 0:
            return None
        item = self._items.get(key)
        if item is None:
            return None
        stamp, value = item
        if time.monotonic() - stamp > self.ttl:
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: tuple[Any, ...], value: SearchResponse) -> None:
        if self.size == 0 or self.ttl == 0:
            return
        self._items[key] = (time.monotonic(), value)
        self._items.move_to_end(key)
        while len(self._items) > self.size:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()


class Retriever:
    def __init__(
        self,
        settings: Settings,
        store: MilvusStore,
        embedder: Embedder,
        reranker: CrossEncoderReranker | None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self._cache = _TTLCache(settings.cache_size, float(settings.cache_ttl_seconds))

    def invalidate(self) -> None:
        """Bir repo yeniden indexlenince eski cevaplar eski satırları gösterir; hepsini at."""
        self._cache.clear()

    def search(self, request: SearchRequest) -> SearchResponse:
        key = request.cache_key()
        cached = self._cache.get(key)
        if cached is not None:
            return SearchResponse(
                hits=[_copy(hit) for hit in cached.hits],
                mode=cached.mode,
                reranked=cached.reranked,
                candidates=cached.candidates,
                timings_ms=dict(cached.timings_ms),
                cached=True,
                weak_match=cached.weak_match,
                dropped=cached.dropped,
            )
        response = self._search(request)
        self._cache.set(key, response)
        return response

    def _search(self, request: SearchRequest) -> SearchResponse:
        settings = self.settings
        started = time.perf_counter()
        timings: dict[str, float] = {}
        query = request.query.strip()
        top_k = request.k or settings.top_k
        use_rerank = (
            settings.rerank_enabled if request.rerank is None else request.rerank
        ) and self.reranker is not None
        depth = max(request.candidates or settings.candidates, top_k) if use_rerank else top_k
        mode = _resolve_mode(request.mode or settings.search_mode, settings.prose_mode, query)
        expression = build_filter(
            request.repo_ids or None,
            request.path_prefix or None,
            request.lang or None,
            request.category or None,
        )

        dense_hits: list[Hit] = []
        bm25_hits: list[Hit] = []
        if mode in ("dense", "hybrid"):
            mark = time.perf_counter()
            vector = self.embedder.encode_one(query)
            timings["embed"] = _ms(mark)
            mark = time.perf_counter()
            dense_hits = self.store.dense_search(vector, depth, expression)
            timings["dense"] = _ms(mark)
        if mode in ("bm25", "hybrid"):
            mark = time.perf_counter()
            bm25_hits = self.store.bm25_search(query, depth, expression)
            timings["bm25"] = _ms(mark)

        if mode == "hybrid":
            candidates = fuse_rrf(dense_hits, bm25_hits, settings.rrf_k)[:depth]
        else:
            candidates = dense_hits or bm25_hits

        if use_rerank and self.reranker is not None and candidates:
            mark = time.perf_counter()
            hits = self.reranker.rerank(query, candidates, top_k)
            timings["rerank"] = _ms(mark)
        else:
            hits = candidates[:top_k]

        hits, dropped = apply_floor(mode, hits, settings.min_dense_score)
        for rank, hit in enumerate(hits):
            hit.rank = rank

        timings["total"] = _ms(started)
        return SearchResponse(
            hits=hits,
            mode=mode,
            reranked=bool(use_rerank and candidates),
            candidates=len(candidates),
            timings_ms=timings,
            weak_match=is_weak_match(mode, hits, settings.weak_dense_score),
            dropped=dropped,
        )


def apply_floor(mode: str, hits: Sequence[Hit], floor: float) -> tuple[list[Hit], int]:
    """Sert taban: dense skoru tabanın altındaki parça atılır. BM25 skoru sınırsız
    olduğundan sembol aramasına dokunulmaz; hybrid'de yalnız BM25'ten gelen (dense skoru
    olmayan) parça tam kelime eşleşmesidir, kalır."""
    if mode == "bm25" or floor <= 0:
        return list(hits), 0
    kept = [hit for hit in hits if hit.scores.get("dense", 1.0) >= floor]
    return kept, len(hits) - len(kept)


def is_weak_match(mode: str, hits: Sequence[Hit], floor: float) -> bool:
    """Boş sonuç zayıf değil, "yok"tur; BM25 skoru sınırsız olduğundan sembol
    aramasında karar verilmez. Dense/hybrid'de en iyi dense skoru eşiğin altındaysa zayıf."""
    if not hits or mode == "bm25":
        return False
    best = max((hit.scores.get("dense", 0.0) for hit in hits), default=0.0)
    return best < floor


def _resolve_mode(mode: str, prose_mode: str, query: str) -> str:
    if mode != "auto":
        return mode
    return "bm25" if looks_like_symbol(query) else prose_mode


def fuse_rrf(dense: Sequence[Hit], bm25: Sequence[Hit], k: int = 60) -> list[Hit]:
    """Reciprocal rank fusion; kanal skorları korunur, `rrf` eklenir."""
    merged: dict[str, Hit] = {}
    for channel_hits in (dense, bm25):
        for rank, hit in enumerate(channel_hits):
            existing = merged.get(hit.id)
            if existing is None:
                existing = _copy(hit)
                existing.scores = dict(hit.scores)
                merged[hit.id] = existing
            else:
                existing.scores.update(hit.scores)
            existing.scores["rrf"] = existing.scores.get("rrf", 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(merged.values(), key=lambda hit: -hit.scores["rrf"])
    for rank, hit in enumerate(fused):
        hit.rank = rank
    return fused


def _copy(hit: Hit) -> Hit:
    return Hit(
        id=hit.id,
        repo_id=hit.repo_id,
        path=hit.path,
        symbol=hit.symbol,
        parent_symbol=hit.parent_symbol,
        kind=hit.kind,
        lang=hit.lang,
        category=hit.category,
        start_line=hit.start_line,
        end_line=hit.end_line,
        content=hit.content,
        context=hit.context,
        scores=dict(hit.scores),
        rank=hit.rank,
    )


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000
