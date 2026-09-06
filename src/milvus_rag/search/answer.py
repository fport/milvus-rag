"""Turns retriever results into a cited answer.

Line numbers go into the prompt and every claim is required to carry an [n]
citation: a sentence without one is suspect, and hallucination becomes visible.
If the chunks do not answer the question the model says so; it does not invent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from milvus_rag.llm import LLM
from milvus_rag.models import Hit
from milvus_rag.search.retrieve import Retriever, SearchRequest, SearchResponse

SYSTEM_PROMPT = """You are a technical assistant who knows a software team's codebase.
You will be given a question and numbered code chunks taken from that codebase.

Rules:
1. Answer ONLY from the chunks given. Anything not in them, you do not know.
2. Mark every claim with the number of the chunk it rests on: [1], [2]. Never write an
   uncited claim.
3. If the chunks do not answer the question, say so plainly, and suggest which file to
   look at if the chunks let you infer it. Do not invent.
4. Chunks marked DOCUMENT are plan/design text: the files, functions and code named in
   them may not exist in the code. Do not present those as "the code says"; say
   "according to the document".
5. Write file paths, function, type and variable names verbatim; never translate them.
6. Answer in the language the question was asked in. Be short and direct; quote code
   where it helps."""

WEAK_NOTE = (
    "Note: the best match is weak; chunks at this level are often irrelevant. "
    "If they do not answer the question, say so — do not force it."
)


@dataclass(slots=True)
class AnswerResult:
    answer: str
    sources: list[dict[str, Any]]
    provider: str
    model: str
    elapsed_ms: float
    search: SearchResponse
    prompt_chars: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "provider": self.provider,
            "model": self.model,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "prompt_chars": self.prompt_chars,
            "search": {
                "mode": self.search.mode,
                "reranked": self.search.reranked,
                "candidates": self.search.candidates,
                "timings_ms": {k: round(v, 1) for k, v in self.search.timings_ms.items()},
                "cached": self.search.cached,
                "weak_match": self.search.weak_match,
            },
        }


def build_prompt(question: str, hits: list[Hit], weak_match: bool = False) -> str:
    """The single-request path: no agent loop, the signals go into the prompt.

    Document chunks are labelled (so code inside a plan is not mistaken for real code)
    and a weak match gets a note, leaving "I could not find it" available.
    """
    if not hits:
        return f"(no code chunks were found)\n\nQuestion: {question}"
    blocks = []
    for index, hit in enumerate(hits, start=1):
        where = f"{hit.path}:{hit.start_line}-{hit.end_line}"
        label = " — DOCUMENT" if hit.category == "doc" else ""
        title = f" — {hit.kind} {hit.symbol}" if hit.symbol else ""
        context = f"\n{hit.context}" if hit.context else ""
        blocks.append(
            f"[{index}] {where}{label}{title} (repo: {hit.repo_id}){context}\n"
            f"```{hit.lang or ''}\n{hit.content}\n```"
        )
    note = f"\n\n{WEAK_NOTE}" if weak_match else ""
    return (
        "CODE CHUNKS (most relevant first):\n\n"
        + "\n\n".join(blocks)
        + note
        + f"\n\nQuestion: {question}"
    )


def ask(retriever: Retriever, llm: LLM, request: SearchRequest) -> AnswerResult:
    started = time.perf_counter()
    search = retriever.search(request)
    prompt = build_prompt(request.query, search.hits, search.weak_match)
    answer = llm.complete(SYSTEM_PROMPT, prompt, max_tokens=8192)
    return AnswerResult(
        answer=answer,
        sources=[
            {
                "n": index,
                "repo_id": hit.repo_id,
                "path": hit.path,
                "start_line": hit.start_line,
                "end_line": hit.end_line,
                "symbol": hit.symbol,
                "kind": hit.kind,
                "category": hit.category,
                "scores": {k: round(v, 6) for k, v in hit.scores.items()},
            }
            for index, hit in enumerate(search.hits, start=1)
        ],
        provider=llm.provider,
        model=llm.model,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        search=search,
        prompt_chars=len(prompt),
    )
