"""Retriever sonuçlarını atıflı bir cevaba çevirir.

Satır numaraları prompt'a girer ve her iddia için [n] atıf istenir: atıfsız
cümle şüphelidir, halüsinasyon görünür olur. Parçalarda cevap yoksa model
"bulamadım" der; uydurmaz.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from milvus_rag.llm import LLM
from milvus_rag.models import Hit
from milvus_rag.search.retrieve import Retriever, SearchRequest, SearchResponse

SYSTEM_PROMPT = """Sen bir yazılım ekibinin kod tabanını bilen teknik asistansın.
Sana bir soru ve o kod tabanından alınmış numaralı kod parçaları verilecek.

Kurallar:
1. YALNIZCA verilen parçalara dayanarak cevapla. Parçalarda olmayan bir şeyi bilmiyorsun.
2. Her iddianı dayandığı parçanın numarasıyla işaretle: [1], [2] gibi. Atıfsız iddia yazma.
3. Parçalar soruyu cevaplamıyorsa bunu açıkça söyle ve hangi dosyaya bakılabileceğini
   parçalardan çıkarabiliyorsan öner. Uydurma.
4. DOKÜMAN işaretli parçalar plan/tasarım metnidir: içinde geçen dosya, fonksiyon ve kod
   kodda var olmayabilir. Onları "kodda böyle" diye sunma; "dokümana göre" de.
5. Dosya yolu, fonksiyon, tip ve değişken adlarını aynen yaz, çevirme.
6. Soru hangi dildeyse o dilde cevapla. Kısa ve doğrudan ol; gerekirse kod alıntıla."""

WEAK_NOTE = (
    "Not: en iyi eşleşme zayıf; bu seviyedeki parçalar sık sık alakasız çıkıyor. "
    "Soruyu cevaplamıyorlarsa bunu söyle, zorlama."
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
    """Tek istekte cevap üreten yol: ajan döngüsü yok, sinyaller prompt'a girer.

    Doküman parçası etiketlenir (plan metnindeki kod gerçek sanılmasın), zayıf
    eşleşme not düşülür ("bulamadım" demek serbest olsun).
    """
    if not hits:
        return f"(hiç kod parçası bulunamadı)\n\nSoru: {question}"
    blocks = []
    for index, hit in enumerate(hits, start=1):
        where = f"{hit.path}:{hit.start_line}-{hit.end_line}"
        label = " — DOKÜMAN" if hit.category == "doc" else ""
        title = f" — {hit.kind} {hit.symbol}" if hit.symbol else ""
        context = f"\n{hit.context}" if hit.context else ""
        blocks.append(
            f"[{index}] {where}{label}{title} (repo: {hit.repo_id}){context}\n"
            f"```{hit.lang or ''}\n{hit.content}\n```"
        )
    note = f"\n\n{WEAK_NOTE}" if weak_match else ""
    return (
        "KOD PARÇALARI (en alakalı önce):\n\n"
        + "\n\n".join(blocks)
        + note
        + f"\n\nSoru: {question}"
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
