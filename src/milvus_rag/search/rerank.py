"""Cross-encoder reranker.

Bi-encoder soruyu ve chunk'ı ayrı ayrı vektöre çevirir: ucuz, kaba. Cross-encoder
ikisini yan yana okur ve "bu chunk bu soruya cevap mı" diye puanlar: isabetli
ama her çift için ayrı forward pass. O yüzden zincir: 40-100 aday → reranker → 8.
Reranker'a 8 aday vermek sadece sıralar, recall'a dokunamaz.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from milvus_rag.log import get_logger
from milvus_rag.models import Hit

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

log = get_logger("rerank")


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_tokens: int = 1024, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            started = time.perf_counter()
            # fp16, MPS/CUDA'da 2.2x hız (ölçüldü: 40 çift 5.5s → 2.5s), sıralama aynı.
            kwargs = (
                {"model_kwargs": {"torch_dtype": torch.float16}}
                if (torch.backends.mps.is_available() or torch.cuda.is_available())
                else {}
            )
            self._model = CrossEncoder(self.model_name, max_length=self.max_tokens, **kwargs)
            log.info(
                "reranker yüklendi",
                model=self.model_name,
                seconds=round(time.perf_counter() - started, 1),
            )
        return self._model

    def warm_up(self) -> None:
        self.model.predict([("warm", "up")], show_progress_bar=False)

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        if len(hits) == 1:
            hits[0].scores["rerank"] = 1.0
            return list(hits[:top_k])
        # Yol adı da çifte girer: "auth.service.ts içinde createSession" gibi
        # bir soruda dosya adı sinyalin yarısıdır.
        pairs = [(query, f"{hit.path}\n{hit.content}") for hit in hits]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        for hit, score in zip(hits, scores, strict=True):
            hit.scores["rerank"] = float(score)
        ordered = sorted(hits, key=lambda hit: -hit.scores["rerank"])[:top_k]
        for rank, hit in enumerate(ordered):
            hit.rank = rank
        return ordered
