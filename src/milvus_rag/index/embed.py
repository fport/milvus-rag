"""Text → vector.

BGE-M3 by default (1024 dimensions, multilingual): a question in one language and
code in another land in the same space. Vectors are L2-normalized on the way out and
searched with COSINE — mixing metrics silently produces wrong ordering, so it is
done in one place.
The model is loaded once, not per request (10-20 s the first time). Imports live
inside the functions so that importing `milvus_rag` does not pull in torch.
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
        """(n, dimension) float32, rows of unit length."""

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
    """A local model via sentence-transformers."""

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
            # sentence-transformers 5.x renamed it; the old name is gone in new versions.
            get_dimension = getattr(
                model, "get_embedding_dimension", model.get_sentence_embedding_dimension
            )
            dimension = get_dimension()
            if dimension != self.expected_dimension:
                msg = (
                    f"{self.model_name} produces {dimension}-dimensional vectors, but the "
                    f"settings say {self.expected_dimension}; the Milvus schema and the "
                    "embedder must agree"
                )
                raise ValueError(msg)
            log.info(
                "embedding model loaded",
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
    """text-embedding-3-* ; `dimensions` reduces it to 1024, the Milvus schema is unchanged."""

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
        msg = "OpenAI embeddings: out of retries"
        raise RuntimeError(msg)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "openai":
        if not settings.openai_api_key:
            msg = "RAG_EMBEDDING_BACKEND=openai requires OPENAI_API_KEY"
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
