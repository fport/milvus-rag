"""The golden-set runner: Recall@k, MRR, latency; broken down by question kind.

One rule: never ship what you did not measure. Every retrieval change enters this
table as a row, not as "it feels better".

A golden row (JSONL):
  {"q": "where is the JWT issued?", "expect": ["src/auth/session.ts::createSession"],
   "mode": "any", "kind": "prose"}

An `expect` entry is `path` or `path::symbol`. No line numbers: lines shift when code
changes, symbols do not. `mode: any` → one of the listed entries is enough;
`all` (the default) → all of them must come back.

Negative case: `expect: []` — a question whose answer is NOT in the code. The retriever
returns something for every query; what is measured is whether it showed that: an empty
result or a `weak_match` signal = abstain. The rate at which the same signal fires
wrongly on positives (`false_weak`) is reported next to it; the signal only means
something when the two are read together.
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

    @property
    def negative(self) -> bool:
        return not self.expect


@dataclass(slots=True)
class CaseResult:
    question: str
    kind: str
    recall: float
    reciprocal_rank: float
    latency_ms: float
    found: list[str]
    top: list[str]
    negative: bool = False
    weak: bool = False
    # For a negative case: did the system show "no answer" (an empty result or weak_match).
    abstained: bool = False
    # The best dense score among the returned hits: the raw material for recalibrating the
    # thresholds (`min_dense_score`, `weak_dense_score`) — the min of the positives against
    # the max of the negatives.
    top_dense: float | None = None


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
    # The abstain rate on negative cases (None when there are none); and the rate at which
    # the weak-match signal fires wrongly on positives.
    abstain_rate: float | None = None
    false_weak_rate: float = 0.0
    # Calibration summary: the lowest top-dense among the positives that were found, and the
    # highest among the negatives. The floor goes below the first, the note between the two.
    calibration: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "k": self.k,
            "n": self.n,
            "recall@k": round(self.recall_at_k, 3),
            "mrr": round(self.mrr, 3),
            "abstain_rate": None if self.abstain_rate is None else round(self.abstain_rate, 3),
            "false_weak_rate": round(self.false_weak_rate, 3),
            "calibration": self.calibration,
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
                    "weak": result.weak,
                    "top_dense": None if result.top_dense is None else round(result.top_dense, 3),
                    **({"abstained": result.abstained} if result.negative else {}),
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
        expect = tuple(str(item) for item in row["expect"])
        cases.append(
            Case(
                question=str(row["q"]),
                expect=expect,
                mode=str(row.get("mode", "all")),
                kind=str(row.get("kind", "prose" if expect else "negative")),
            )
        )
    return cases


def matches(expected: str, hit: Hit) -> bool:
    """`path` → a file match; `path::symbol` → the symbol, or the one containing it."""
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
    """(recall, reciprocal rank, found). Ranks are deduplicated per file/symbol."""
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
        abstained = not response.hits or response.weak_match
        if case.negative:
            recall, reciprocal, found = float(abstained), 0.0, list[str]()
        else:
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
                negative=case.negative,
                weak=response.weak_match,
                abstained=abstained,
                top_dense=max(
                    (hit.scores["dense"] for hit in response.hits if "dense" in hit.scores),
                    default=None,
                ),
            )
        )

    positives = [result for result in results if not result.negative]
    negatives = [result for result in results if result.negative]
    latencies = sorted(result.latency_ms for result in results) or [0.0]
    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({result.kind for result in results}):
        subset = [result for result in results if result.kind == kind]
        if all(result.negative for result in subset):
            by_kind[kind] = {
                "n": len(subset),
                "abstain": round(statistics.mean(float(r.abstained) for r in subset), 3),
            }
            continue
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
        recall_at_k=statistics.mean(r.recall for r in positives) if positives else 0.0,
        mrr=statistics.mean(r.reciprocal_rank for r in positives) if positives else 0.0,
        p50_ms=latencies[len(latencies) // 2],
        p95_ms=latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        by_kind=by_kind,
        # Not found on a positive + not abstained on a negative: both count as a "miss".
        misses=[result for result in results if result.recall < 1.0],
        abstain_rate=(
            statistics.mean(float(r.abstained) for r in negatives) if negatives else None
        ),
        false_weak_rate=(statistics.mean(float(r.weak) for r in positives) if positives else 0.0),
        calibration=_calibration(positives, negatives),
        config={
            "mode": mode or settings.search_mode,
            "prose_mode": settings.prose_mode,
            "rerank": settings.rerank_enabled if rerank is None else rerank,
            "candidates": candidates or settings.candidates,
            "embedding": settings.embedding_model,
            "rerank_model": settings.rerank_model if settings.rerank_enabled else None,
            "rrf_k": settings.rrf_k,
            "chunk": f"{settings.chunk_min_bytes}-{settings.chunk_max_bytes}",
            "weak_dense_score": settings.weak_dense_score,
            "repo_ids": list(repo_ids),
        },
        results=results,
    )


def _calibration(
    positives: list[CaseResult], negatives: list[CaseResult]
) -> dict[str, float | None]:
    found = [r.top_dense for r in positives if r.recall >= 1.0 and r.top_dense is not None]
    junk = [r.top_dense for r in negatives if r.top_dense is not None]
    return {
        "positive_found_min_top_dense": round(min(found), 3) if found else None,
        "positive_found_median_top_dense": round(statistics.median(found), 3) if found else None,
        "negative_max_top_dense": round(max(junk), 3) if junk else None,
        "negative_median_top_dense": round(statistics.median(junk), 3) if junk else None,
    }


def save_report(report: EvalReport, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M")
    target = results_dir / f"{stamp}_{report.tag}.json"
    target.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target
