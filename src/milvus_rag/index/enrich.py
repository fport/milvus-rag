"""Optional: a 2-3 sentence description of every chunk, from the LLM.

A piece of code contains almost no natural language, and what it does contain is in
whatever language its author wrote comments in. Measured in an earlier experiment:
plain-language questions in one language, over code in another, scored recall@5 = 0.04.
The cure is not a better matcher, it is writing the missing prose (Anthropic's
"contextual retrieval": a 35% drop in retrieval failures).

Because BGE-M3 is multilingual this layer is off by default; measure with `rag eval`
before turning it on. When it is on, descriptions are cached by chunk hash, so
re-indexing does not pay the same bill twice.

The output language is `RAG_ENRICH_LANGUAGE`: the gain comes from adding prose in the
language questions are asked in, so it should match your question traffic. It is part of
the cache key — changing it regenerates descriptions rather than serving stale ones.

One call per file (the whole file + numbered chunks). The answer goes through a coverage
check: a 7B model will happily return one object and drop the rest in silence. Drifting
into CJK, Hangul or Cyrillic is rejected — qwen2.5:7b slides into Chinese mid-sentence
under load and does not mention it. (If you set a language written in one of those
scripts, that guard is the thing to adjust.)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

from milvus_rag.db import Database
from milvus_rag.llm import LLM, LLMError
from milvus_rag.log import get_logger
from milvus_rag.models import ChunkRecord

log = get_logger("enrich")

FILE_CONTEXT_CHARS = 8_000

SYSTEM_PROMPT_TEMPLATE = """You are a technical writer documenting a codebase in {language}.
You will be given a whole file and numbered chunks taken from that file.
For EVERY CHUNK, say in 2-3 sentences, in {language}, what the code in that chunk does.

Answer with this JSON schema and nothing else:
{{"chunks": [{{"n": <chunk number>, "description": "<2-3 sentences in {language}>"}}]}}

Hard rules:
- Write one entry for EVERY chunk you are given. Skip none.
- Describe only the code in that chunk, not the rest of the file.
- Do NOT translate function, variable, type or table names; keep them verbatim.
- Do not open with "This chunk" or "This file"; say directly what the code does.
- Never mention a function that is not in the chunk. If unsure, write less; invent nothing."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_DRIFT_SCRIPT = re.compile(r"[぀-ヿ一-鿿가-힯Ѐ-ӿ]")


def chunk_hash(record: ChunkRecord) -> str:
    return hashlib.sha256(f"{record.header}\n{record.text}".encode()).hexdigest()


def is_usable(description: str) -> bool:
    text = description.strip()
    return len(text) >= 20 and not _DRIFT_SCRIPT.search(text)


class Enricher:
    def __init__(
        self, llm: LLM, db: Database, batch_chunks: int = 6, language: str = "English"
    ) -> None:
        self.llm = llm
        self.db = db
        self.batch_chunks = batch_chunks
        self.language = language
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=language)
        self.generated = 0
        self.cached = 0
        self.rejected = 0
        self.failures = 0

    @property
    def cache_key(self) -> str:
        # The language is part of the key: switching it must regenerate, not serve stale text.
        return f"{self.llm.model}:{self.language}"

    def describe(self, path: str, text: str, records: Sequence[ChunkRecord]) -> list[str]:
        """One description per record; an empty string where none could be produced."""
        if not records:
            return []
        hashes = [chunk_hash(record) for record in records]
        known = self.db.get_enrichments(hashes, self.cache_key)
        self.cached += len(known)
        descriptions = [known.get(digest, "") for digest in hashes]
        missing = [index for index, digest in enumerate(hashes) if digest not in known]
        if not missing:
            return descriptions

        fresh: list[tuple[str, str]] = []
        for start in range(0, len(missing), self.batch_chunks):
            batch = missing[start : start + self.batch_chunks]
            answers = self._describe_batch(path, text, records, batch)
            for index, description in zip(batch, answers, strict=True):
                if description:
                    descriptions[index] = description
                    fresh.append((hashes[index], description))
        if fresh:
            self.db.set_enrichments(fresh, self.cache_key)
        return descriptions

    def _describe_batch(
        self, path: str, text: str, records: Sequence[ChunkRecord], indices: list[int]
    ) -> list[str]:
        parts = "\n\n".join(
            f"### Chunk {number}\n{records[index].text}" for number, index in enumerate(indices, 1)
        )
        user = (
            f"File: {path}\n\n```\n{text[:FILE_CONTEXT_CHARS]}\n```\n\nNumbered chunks:\n\n{parts}"
        )
        try:
            raw = self.llm.complete(self.system_prompt, user, max_tokens=2048)
        except LLMError as error:
            self.failures += len(indices)
            log.warning("enrichment call failed", path=path, error=str(error))
            return [""] * len(indices)

        parsed = _parse(raw)
        results: list[str] = []
        for number, _ in enumerate(indices, 1):
            description = parsed.get(number, "").strip()
            if description and is_usable(description):
                results.append(description)
                self.generated += 1
            else:
                results.append("")
                if description:
                    self.rejected += 1
                else:
                    self.failures += 1
        return results


def _parse(raw: str) -> dict[int, str]:
    match = _JSON_OBJECT.search(raw or "")
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    items = payload.get("chunks") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return {}
    found: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_number = item.get("n")
        if raw_number is None:
            continue
        try:
            number = int(raw_number)
        except (TypeError, ValueError):
            continue
        found[number] = str(item.get("description") or "")
    return found
