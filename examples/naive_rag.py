"""Rung 1 of the RAG ladder: the whole thing in one file, so its limits are visible.

Character windows, one embedding model, cosine top-k, no database. Roughly what
every "build a RAG in 50 lines" tutorial produces, and it genuinely works — on the
right question, against a small corpus.

It is here to be run, not read: the four failure modes the rest of this project
exists to fix are easier to believe after watching them happen.

    uv run python examples/naive_rag.py ~/code/my-api "how is the queue drained?"
    uv run python examples/naive_rag.py ~/code/my-api --failures

`--failures` runs the four questions that break it:

  1. an exact symbol       — lexical identity is invisible to an embedding
  2. a question whose answer straddles a window boundary
  3. a question in a language the code is not commented in
  4. a question the corpus cannot answer — and it answers anyway

Everything here is deliberately naive. Nothing in this file is imported by the
service; `milvus_rag.index` is the version that is not.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The service's own filters and embedder, so the comparison is about retrieval and
# not about which files each side happened to read.
from milvus_rag.config import Settings
from milvus_rag.index.embed import Embedder, build_embedder
from milvus_rag.sources.files import iter_source_files

WINDOW_CHARACTERS = 1000
OVERLAP_CHARACTERS = 200


@dataclass(frozen=True, slots=True)
class Window:
    """A slice of a file. Note what it does not have: a symbol name.

    Naive chunking cannot tell you what a window *is*, only where it starts. Every
    citation this script prints is therefore `path:offset`, which is not something a
    reader can act on.
    """

    path: str
    start: int
    text: str

    @property
    def ref(self) -> str:
        return f"{self.path}:{self.start}"


def split(
    text: str, *, size: int = WINDOW_CHARACTERS, overlap: int = OVERLAP_CHARACTERS
) -> list[tuple[int, str]]:
    """Fixed windows with a fixed overlap. The overlap is the tell.

    It exists because the boundary is arbitrary and lands mid-function often enough
    to matter; the fix is to make the boundary meaningful instead, which needs a
    parser (rung 2).
    """
    if size <= overlap:
        msg = "the overlap has to be smaller than the window"
        raise ValueError(msg)
    step = size - overlap
    return [(start, text[start : start + size]) for start in range(0, max(len(text), 1), step)]


def build_index(
    root: Path, settings: Settings, embedder: Embedder
) -> tuple[list[Window], np.ndarray]:
    """Read, window, embed. Everything lives in memory and dies with the process.

    That is the honest shape of rung 1: there is no incremental update, because
    there is nothing to update — the next run re-embeds the entire corpus.
    """
    windows: list[Window] = []
    for source in iter_source_files(root, settings.max_file_bytes):
        for start, chunk in split(source.text):
            if chunk.strip():
                windows.append(Window(source.path, start, chunk))

    if not windows:
        msg = f"no indexable files under {root}"
        raise SystemExit(msg)

    print(f"embedding {len(windows)} windows with {embedder.name} …", file=sys.stderr)
    vectors = embedder.encode([window.text for window in windows])
    return windows, np.asarray(vectors, dtype=np.float32)


def search(
    query: str, windows: list[Window], vectors: np.ndarray, embedder: Embedder, k: int
) -> list[tuple[float, Window]]:
    """Cosine top-k, and nothing else.

    The vectors are unit length, so a dot product IS the cosine. There is no floor:
    `argsort` always returns k rows, however far away they are. A kNN search means
    "the nearest k" and has no way to express "nothing is near" — which is exactly
    where a downstream LLM starts inventing.
    """
    query_vector = embedder.encode_one(query)
    scores = vectors @ query_vector
    top = np.argsort(-scores)[:k]
    return [(float(scores[index]), windows[index]) for index in top]


def report(query: str, hits: list[tuple[float, Window]]) -> None:
    print(f"\n\033[1m{query}\033[0m")
    for rank, (score, window) in enumerate(hits, start=1):
        first_line = next((line for line in window.text.splitlines() if line.strip()), "")
        print(f"  {rank}. {score:.3f}  {window.ref}")
        print(f"     {first_line.strip()[:90]}")


FAILING_QUESTIONS = [
    ("an exact symbol", "iter_source_files"),
    ("an answer that straddles a boundary", "what happens after the vectors are written?"),
    ("a question in another language", "kuyruk nasil bosaltiliyor?"),
    ("nothing in the corpus answers this", "have you ever been to a five-star resort?"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rung 1 of the RAG ladder: windows, cosine, top-k."
    )
    parser.add_argument("root", type=Path, help="directory to index")
    parser.add_argument("query", nargs="?", help="what to ask")
    parser.add_argument("-k", type=int, default=5, help="results to show")
    parser.add_argument(
        "--failures", action="store_true", help="run the four questions that break it"
    )
    args = parser.parse_args()

    if not args.query and not args.failures:
        parser.error("give a query, or --failures")

    settings = Settings()
    embedder = build_embedder(settings)
    windows, vectors = build_index(args.root, settings, embedder)

    if args.failures:
        for label, question in FAILING_QUESTIONS:
            report(
                f"{question}   \033[2m({label})\033[0m",
                search(question, windows, vectors, embedder, args.k),
            )
        print(
            "\nEvery one of those returned k results with a straight face. "
            "Read the scores: that is the whole problem."
        )
        return

    report(args.query, search(args.query, windows, vectors, embedder, args.k))


if __name__ == "__main__":
    main()
