"""Golden set runner: Recall@k, MRR, gecikme; soru türüne göre kırılım.

Tek kural: ölçmediğin hiçbir şeyi ekleme. Her retrieval değişikliği bu tabloya
bir satır olarak girer, "sanki iyi oldu" olarak değil.

Golden satırı (JSONL):
  {"q": "JWT nerede üretiliyor?", "expect": ["src/auth/session.ts::createSession"],
   "mode": "any", "kind": "prose"}

`expect` girdisi `path` ya da `path::symbol`. Satır numarası yok: kod değişince
satırlar kayar, semboller kalır. `mode: any` → listedekilerden biri yeter;
`all` (varsayılan) → hepsi gelmeli.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from milvus_rag.models import Hit
from milvus_rag.search.retrieve import Retriever, SearchRequest


@dataclass(frozen=True, slots=True)
class Case:
    question: str
    expect: tuple[str, ...]
    mode: str = "all"
    kind: str = "prose"


@dataclass(slots=True)
class CaseResult:
    question: str
    kind: str
    recall: float
    reciprocal_rank: float
    latency_ms: float
    found: list[str]
    top: list[str]


@dataclass(slots=True)
class EvalReport:
    tag: str
    k: int
    n: int
    recall_at_k: float
    mrr: float
    p50_ms: float
    p95_ms: float
    by_kind: dict[str, dict[str, float]]
    misses: list[CaseResult]
    config: dict[str, Any] = field(default_factory=dict)
    results: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "k": self.k,
            "n": self.n,
            "recall@k": round(self.recall_at_k, 3),
            "mrr": round(self.mrr, 3),
            "p50_ms": round(self.p50_ms),
            "p95_ms": round(self.p95_ms),
            "by_kind": self.by_kind,
            "config": self.config,
            "misses": [
                {"q": miss.question, "kind": miss.kind, "top": miss.top[:5]} for miss in self.misses
            ],
            "results": [
                {
                    "q": result.question,
                    "kind": result.kind,
                    "recall": round(result.recall, 3),
                    "rr": round(result.reciprocal_rank, 3),
                    "ms": round(result.latency_ms),
                    "found": result.found,
                }
                for result in self.results
            ],
        }


def load_golden(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        cases.append(
            Case(
                question=str(row["q"]),
                expect=tuple(str(item) for item in row["expect"]),
                mode=str(row.get("mode", "all")),
                kind=str(row.get("kind", "prose")),
            )
        )
    return cases


def matches(expected: str, hit: Hit) -> bool:
    """`path` → dosya eşleşmesi; `path::symbol` → sembol ya da kapsayan sembol."""
    if "::" not in expected:
        return hit.path == expected
    path, symbol = expected.split("::", 1)
    if hit.path != path:
        return False
    return (
        hit.symbol == symbol
        or hit.parent_symbol == symbol
        or hit.parent_symbol.endswith(f".{symbol}")
    )


def score_case(case: Case, hits: list[Hit]) -> tuple[float, float, list[str]]:
    """(recall, reciprocal rank, bulunanlar). Sıra dosya/sembol bazında tekilleştirilmiş."""
    found: dict[str, int] = {}
    for rank, hit in enumerate(hits):
        for expected in case.expect:
            if expected not in found and matches(expected, hit):
                found[expected] = rank
    need = 1 if case.mode == "any" else len(case.expect)
    recall = min(len(found), need) / need if need else 0.0
    reciprocal = 1.0 / (min(found.values()) + 1) if found else 0.0
    return recall, reciprocal, list(found)


def run_eval(
    retriever: Retriever,
    cases: list[Case],
    repo_ids: tuple[str, ...],
    k: int,
    tag: str,
    mode: str | None = None,
    rerank: bool | None = None,
    candidates: int | None = None,
) -> EvalReport:
    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        response = retriever.search(
            SearchRequest(
                query=case.question,
                repo_ids=repo_ids,
                k=k,
                mode=mode,  # type: ignore[arg-type]
                rerank=rerank,
                candidates=candidates,
            )
        )
        latency = (time.perf_counter() - started) * 1000
        recall, reciprocal, found = score_case(case, response.hits)
        results.append(
            CaseResult(
                question=case.question,
                kind=case.kind,
                recall=recall,
                reciprocal_rank=reciprocal,
                latency_ms=latency,
                found=found,
                top=[hit.ref for hit in response.hits[:k]],
            )
        )

    latencies = sorted(result.latency_ms for result in results) or [0.0]
    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({result.kind for result in results}):
        subset = [result for result in results if result.kind == kind]
        by_kind[kind] = {
            "n": len(subset),
            "recall@k": round(statistics.mean(r.recall for r in subset), 3),
            "mrr": round(statistics.mean(r.reciprocal_rank for r in subset), 3),
        }
    settings = retriever.settings
    return EvalReport(
        tag=tag,
        k=k,
        n=len(results),
        recall_at_k=statistics.mean(r.recall for r in results) if results else 0.0,
        mrr=statistics.mean(r.reciprocal_rank for r in results) if results else 0.0,
        p50_ms=latencies[len(latencies) // 2],
        p95_ms=latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        by_kind=by_kind,
        misses=[result for result in results if result.recall < 1.0],
        config={
            "mode": mode or settings.search_mode,
            "prose_mode": settings.prose_mode,
            "rerank": settings.rerank_enabled if rerank is None else rerank,
            "candidates": candidates or settings.candidates,
            "embedding": settings.embedding_model,
            "rerank_model": settings.rerank_model if settings.rerank_enabled else None,
            "rrf_k": settings.rrf_k,
            "chunk": f"{settings.chunk_min_bytes}-{settings.chunk_max_bytes}",
            "repo_ids": list(repo_ids),
        },
        results=results,
    )


def save_report(report: EvalReport, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M")
    target = results_dir / f"{stamp}_{report.tag}.json"
    target.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target
