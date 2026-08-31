"""Metin → vektör.

Varsayılan BGE-M3 (1024 boyut, çok dilli): Türkçe soru ile İngilizce kod aynı
uzayda. Vektörler çıkarken L2-normalize edilir; COSINE ile arama yapılır —
metrik karışıklığı sessizce yanlış sıralama üretir, o yüzden tek yer.

Model istek başına değil bir kez yüklenir (ilk yükleme 10-20 sn). Import'lar
fonksiyon içinde: `milvus_rag`'ı import etmek torch'u çekmesin.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from milvus_rag.config import Settings
from milvus_rag.log import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = get_logger("embed")

Vector = np.ndarray


class Embedder(ABC):
    name: str

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> Vector:
        """(n, dimension) float32, satırlar birim uzunlukta."""

    def encode_one(self, text: str) -> Vector:
        return np.asarray(self.encode([text])[0], dtype=np.float32)

    def warm_up(self) -> None:
        self.encode(["warm up"])


def normalise(vectors: Vector) -> Vector:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    unit: Vector = np.asarray(array / norms, dtype=np.float32)
    return unit


class LocalEmbedder(Embedder):
    """sentence-transformers ile yerel model."""

    def __init__(
        self,
        model_name: str,
        expected_dimension: int,
        batch_size: int = 32,
        max_tokens: int = 1024,
        device: str | None = None,
    ) -> None:
        self.name = model_name
        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.device = device
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            started = time.perf_counter()
            model = SentenceTransformer(self.model_name, device=self.device)
            model.max_seq_length = self.max_tokens
            dimension = model.get_sentence_embedding_dimension()
            if dimension != self.expected_dimension:
                msg = (
                    f"{self.model_name} {dimension} boyutlu vektör üretiyor, ayarlarda "
                    f"{self.expected_dimension} var; Milvus şeması ile embedder uyuşmalı"
                )
                raise ValueError(msg)
            log.info(
                "embedding modeli yüklendi",
                model=self.model_name,
                device=str(model.device),
                seconds=round(time.perf_counter() - started, 1),
            )
            self._model = model
        return self._model

    @property
    def dimension(self) -> int:
        return self.expected_dimension

    def encode(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.empty((0, self.expected_dimension), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return normalise(np.asarray(vectors, dtype=np.float32))


class OpenAIEmbedder(Embedder):
    """text-embedding-3-* ; `dimensions` ile 1024'e indirgenir, Milvus şeması değişmez."""

    def __init__(
        self, api_key: str, model_name: str, dimension: int, batch_size: int = 100
    ) -> None:
        self.name = model_name
        self.model_name = model_name
        self._dimension = dimension
        self.batch_size = batch_size
        self.api_key = api_key

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> Vector:
        import httpx

        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        rows: list[list[float]] = []
        with httpx.Client(timeout=60.0) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                payload: dict[str, Any] = {
                    "model": self.model_name,
                    "input": batch,
                    "dimensions": self._dimension,
                }
                rows.extend(self._request(client, payload))
        return normalise(np.asarray(rows, dtype=np.float32))

    def _request(self, client: Any, payload: dict[str, Any]) -> list[list[float]]:
        import httpx

        delay = 1.0
        for attempt in range(4):
            response = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            if response.status_code < 400:
                data = response.json()["data"]
                return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(delay)
                delay *= 2
                continue
            msg = f"OpenAI embeddings {response.status_code}: {response.text[:300]}"
            raise httpx.HTTPStatusError(msg, request=response.request, response=response)
        msg = "OpenAI embeddings: yeniden denemeler tükendi"
        raise RuntimeError(msg)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "openai":
        if not settings.openai_api_key:
            msg = "RAG_EMBEDDING_BACKEND=openai için OPENAI_API_KEY gerekli"
            raise ValueError(msg)
        model = settings.embedding_model
        if model.startswith("BAAI/"):
            model = "text-embedding-3-large"
        return OpenAIEmbedder(settings.openai_api_key, model, settings.embedding_dim)
    return LocalEmbedder(
        model_name=settings.embedding_model,
        expected_dimension=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
        max_tokens=settings.embedding_max_tokens,
        device=settings.embedding_device,
    )
