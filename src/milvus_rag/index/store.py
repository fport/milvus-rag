"""The Milvus store: one collection, `repo_id` as the partition key.

Dense and sparse live in the same collection. Nobody writes the sparse field; Milvus'
own BM25 Function derives it from `indexed_text`, so there is no separate lexical index.

Deletes and updates go through `repo_id + path`: when a file changes, the chunks at
that path are deleted first and the new ones inserted (no archiving — the data in
Milvus is derived, the source is the repo). Thanks to the partition key, a
`repo_id in [...]` filter descends only into the relevant partitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from milvus_rag.log import get_logger
from milvus_rag.models import Hit

if TYPE_CHECKING:
    from pymilvus import MilvusClient

log = get_logger("store")

METRIC = "COSINE"
DENSE_FIELD = "dense"
SPARSE_FIELD = "sparse"
BM25_FUNCTION = "indexed_text_bm25"
HNSW_INDEX = {"M": 16, "efConstruction": 200}
HNSW_SEARCH = {"ef": 128}
CONSISTENCY = "Session"  # read-your-writes: counting right after a delete+write is correct

OUTPUT_FIELDS = [
    "id",
    "repo_id",
    "path",
    "symbol",
    "parent_symbol",
    "kind",
    "lang",
    "category",
    "start_line",
    "end_line",
    "content",
    "context",
]

# The Milvus VARCHAR limit is in bytes; stay well away from the edge.
_MAX_TEXT_BYTES = 60_000


class MilvusStore:
    def __init__(self, uri: str, collection: str, dimension: int) -> None:
        self.uri = uri
        self.collection = collection
        self.dimension = dimension
        self._client: MilvusClient | None = None
        self._ready = False

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self.uri)
        return self._client

    # --------------------------------------------------------------- schema
    def ensure_collection(self) -> None:
        if self._ready:
            return
        if self.client.has_collection(self.collection):
            self.client.load_collection(self.collection)
            self._ready = True
            return

        from pymilvus import DataType, Function, FunctionType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=1024)
        schema.add_field("repo_id", DataType.VARCHAR, max_length=128, is_partition_key=True)
        schema.add_field("path", DataType.VARCHAR, max_length=1024)
        schema.add_field("symbol", DataType.VARCHAR, max_length=512)
        schema.add_field("parent_symbol", DataType.VARCHAR, max_length=512)
        schema.add_field("kind", DataType.VARCHAR, max_length=32)
        schema.add_field("lang", DataType.VARCHAR, max_length=32)
        schema.add_field("category", DataType.VARCHAR, max_length=16)
        schema.add_field("ordinal", DataType.INT32)
        schema.add_field("start_line", DataType.INT32)
        schema.add_field("end_line", DataType.INT32)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("indexed_text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        schema.add_field("context", DataType.VARCHAR, max_length=8192)
        schema.add_field("symbols", DataType.VARCHAR, max_length=2048)
        schema.add_field("file_sha", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("indexed_at", DataType.VARCHAR, max_length=32)
        schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name=BM25_FUNCTION,
                function_type=FunctionType.BM25,
                input_field_names=["indexed_text"],
                output_field_names=[SPARSE_FIELD],
            )
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=DENSE_FIELD, index_type="HNSW", metric_type=METRIC, params=dict(HNSW_INDEX)
        )
        index_params.add_index(
            field_name=SPARSE_FIELD, index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
        )
        self.client.create_collection(
            self.collection,
            schema=schema,
            index_params=index_params,
            consistency_level=CONSISTENCY,
        )
        self.client.load_collection(self.collection)
        log.info("collection created", collection=self.collection, dim=self.dimension)
        self._ready = True

    def healthy(self) -> bool:
        try:
            self.client.list_collections()
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------- write
    def insert(self, rows: Sequence[dict[str, Any]]) -> int:
        if not rows:
            return 0
        self.ensure_collection()
        for row in rows:
            row["content"] = _clip(row["content"])
            row["indexed_text"] = _clip(row["indexed_text"])
            row["context"] = _clip(row.get("context", ""), 8000)
            row["symbols"] = _clip(row.get("symbols", ""), 2000)
        result = self.client.insert(self.collection, data=list(rows))
        return int(result.get("insert_count", len(rows)))

    def delete_paths(self, repo_id: str, paths: Sequence[str]) -> None:
        if not paths or not self.client.has_collection(self.collection):
            return
        self.ensure_collection()
        for start in range(0, len(paths), 200):
            quoted = ", ".join(f'"{_escape(path)}"' for path in paths[start : start + 200])
            self.client.delete(
                self.collection, filter=f'repo_id == "{_escape(repo_id)}" and path in [{quoted}]'
            )

    def delete_repo(self, repo_id: str) -> None:
        if not self.client.has_collection(self.collection):
            return
        self.ensure_collection()
        self.client.delete(self.collection, filter=f'repo_id == "{_escape(repo_id)}"')

    def flush(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.flush(self.collection)

    def drop(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self._ready = False

    # ----------------------------------------------------------------- read
    def count(self, repo_id: str | None = None) -> int:
        if not self.client.has_collection(self.collection):
            return 0
        self.ensure_collection()
        expression = f'repo_id == "{_escape(repo_id)}"' if repo_id else ""
        rows = self.client.query(self.collection, filter=expression, output_fields=["count(*)"])
        return int(rows[0]["count(*)"]) if rows else 0

    def dense_search(self, vector: np.ndarray, limit: int, expression: str = "") -> list[Hit]:
        if not self.client.has_collection(self.collection):
            return []
        self.ensure_collection()
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1).tolist()
        matches = self.client.search(
            self.collection,
            data=query,
            anns_field=DENSE_FIELD,
            limit=limit,
            filter=expression,
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": METRIC, "params": dict(HNSW_SEARCH)},
        )
        return _to_hits(matches, "dense")

    def bm25_search(self, text: str, limit: int, expression: str = "") -> list[Hit]:
        if not text.strip() or not self.client.has_collection(self.collection):
            return []
        self.ensure_collection()
        matches = self.client.search(
            self.collection,
            data=[text],
            anns_field=SPARSE_FIELD,
            limit=limit,
            filter=expression,
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": "BM25"},
        )
        return _to_hits(matches, "bm25")

    def chunks_of(self, repo_id: str, path: str) -> list[Hit]:
        """A file's chunks, in order. For neighbour expansion and for debugging."""
        if not self.client.has_collection(self.collection):
            return []
        self.ensure_collection()
        rows = self.client.query(
            self.collection,
            filter=f'repo_id == "{_escape(repo_id)}" and path == "{_escape(path)}"',
            output_fields=[*OUTPUT_FIELDS, "ordinal"],
        )
        hits = [_to_hit(row, {}) for row in sorted(rows, key=lambda row: int(row["ordinal"]))]
        for rank, hit in enumerate(hits):
            hit.rank = rank
        return hits


def build_filter(
    repo_ids: Sequence[str] | None = None,
    path_prefix: str | None = None,
    lang: str | None = None,
    category: str | None = None,
) -> str:
    clauses: list[str] = []
    if repo_ids:
        quoted = ", ".join(f'"{_escape(repo_id)}"' for repo_id in repo_ids)
        clauses.append(f"repo_id in [{quoted}]")
    if path_prefix:
        clauses.append(f'path like "{_escape(path_prefix)}%"')
    if lang:
        clauses.append(f'lang == "{_escape(lang)}"')
    if category:
        clauses.append(f'category == "{_escape(category)}"')
    return " and ".join(f"({clause})" for clause in clauses)


def _to_hits(matches: Sequence[Sequence[dict[str, Any]]], channel: str) -> list[Hit]:
    rows = matches[0] if matches else []
    hits: list[Hit] = []
    for rank, row in enumerate(rows):
        entity = row.get("entity", row)
        hit = _to_hit(entity, {channel: float(row["distance"])})
        hit.rank = rank
        hits.append(hit)
    return hits


def _to_hit(entity: dict[str, Any], scores: dict[str, float]) -> Hit:
    return Hit(
        id=str(entity["id"]),
        repo_id=str(entity.get("repo_id", "")),
        path=str(entity.get("path", "")),
        symbol=str(entity.get("symbol", "")),
        parent_symbol=str(entity.get("parent_symbol", "")),
        kind=str(entity.get("kind", "")),
        lang=str(entity.get("lang", "")),
        category=str(entity.get("category", "")),
        start_line=int(entity.get("start_line", 0)),
        end_line=int(entity.get("end_line", 0)),
        content=str(entity.get("content", "")),
        context=str(entity.get("context", "")),
        scores=scores,
    )


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _clip(text: str, limit: int = _MAX_TEXT_BYTES) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")
