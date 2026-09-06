"""Cross-encoder reranker.

A bi-encoder turns the question and the chunk into vectors separately: cheap and
coarse. A cross-encoder reads them side by side and scores "does this chunk answer
this question": accurate, but one forward pass per pair. Hence the chain:
40-100 candidates → reranker → 8. Handing the reranker 8 only re-orders them.
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
            # fp16 is 2.2x on MPS/CUDA (measured: 40 pairs 5.5s → 2.5s), same ordering.
            kwargs = (
                {"model_kwargs": {"torch_dtype": torch.float16}}
                if (torch.backends.mps.is_available() or torch.cuda.is_available())
                else {}
            )
            self._model = CrossEncoder(self.model_name, max_length=self.max_tokens, **kwargs)
            log.info(
                "reranker loaded",
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
        # The path goes into the pair too: in a question like "createSession in
        # auth.service.ts" the file name is half the signal.
        pairs = [(query, f"{hit.path}\n{hit.content}") for hit in hits]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        for hit, score in zip(hits, scores, strict=True):
            hit.scores["rerank"] = float(score)
        ordered = sorted(hits, key=lambda hit: -hit.scores["rerank"])[:top_k]
        for rank, hit in enumerate(ordered):
            hit.rank = rank
        return ordered
