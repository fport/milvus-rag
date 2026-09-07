"""The ladder's rung-1 example is documentation that runs; a broken one teaches nothing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_example() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "naive_rag.py"
    spec = importlib.util.spec_from_file_location("naive_rag_example", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example = _load_example()


def test_windows_overlap_and_cover_the_text() -> None:
    text = "".join(f"line {index}\n" for index in range(400))
    windows = example.split(text, size=100, overlap=20)

    assert "".join(chunk for _, chunk in windows).count("line 399") >= 1
    starts = [start for start, _ in windows]
    assert starts == sorted(starts)
    # The overlap is what lets a boundary-straddling answer survive at all — if it
    # ever stopped overlapping, the example would silently stop demonstrating rung 1.
    first, second = windows[0][1], windows[1][1]
    assert first[-20:] == second[:20]


def test_an_overlap_at_least_as_large_as_the_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than the window"):
        example.split("x", size=10, overlap=10)


def test_search_returns_k_hits_however_far_away_they_are() -> None:
    """The point of the example: there is no floor, so a bad query still gets k rows."""

    class _Embedder:
        name = "fake"

        def encode_one(self, text: str) -> np.ndarray:
            return np.asarray([0.0, 1.0], dtype=np.float32)

    windows = [example.Window("a.py", 0, "a"), example.Window("b.py", 0, "b")]
    vectors = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)

    hits = example.search("anything", windows, vectors, _Embedder(), k=2)

    assert len(hits) == 2
    assert hits[0][0] < 0.2  # nothing is anywhere near, and it answers anyway
    assert hits[0][1].ref == "b.py:0"
