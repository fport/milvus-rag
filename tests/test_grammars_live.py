"""Grammar smoke test: every grammar in PREFETCH_GRAMMARS is chunked in its own process,
with degenerate inputs (empty, plain text, 300 KB with many members, unbalanced parens,
random bytes, one very long line). A segfault takes down the subprocess, not the main
one, so when the pack version changes you see exactly which grammar broke.

`RAG_LIVE=1 uv run pytest tests/test_grammars_live.py -q` — downloads the grammars (network).
The extensions of a grammar that crashes or exceeds 300 s move into `CODE_WITHOUT_GRAMMAR`
(that is how sql and cobol got there); a parse cannot be time-limited in this version.
"""

import os
import subprocess
import sys

import pytest

from milvus_rag.sources.files import GRAMMAR_ALIASES, GRAMMAR_BY_FILENAME, PREFETCH_GRAMMARS

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE") != "1", reason="runs with RAG_LIVE=1"),
]

_CHILD = """
import random, sys
from milvus_rag.index.chunk import _parser_for, chunk_file
path, language = sys.argv[1], sys.argv[2]
assert _parser_for(language) is not None, f"{language} yüklenemedi"
random.seed(1)
member = (
    "public static int Foo_{i}(int a, string b) "
    "{{ if (a > 1) {{ return a * 2; }} return b.Length; }}\\n"
)
samples = [
    "",
    "merhaba dünya\\n" * 20,
    "".join(member.format(i=i) for i in range(3000)),
    "{{{{ (( [[ <<\\n" * 500 + "}}}} \\n",
    "".join(chr(random.randint(32, 126)) for _ in range(50_000)),
    "x" * 100_000,
]
print(sum(len(chunk_file(path, text)) for text in samples))
"""


def sample_path_for(grammar: str) -> str:
    for name, candidate in GRAMMAR_BY_FILENAME.items():
        if candidate == grammar:
            return name
    for ext, candidate in GRAMMAR_ALIASES.items():
        if candidate == grammar:
            return f"sample{ext}"
    return f"sample.{grammar}"


@pytest.mark.parametrize("grammar", sorted(PREFETCH_GRAMMARS))
def test_grammar_survives_degenerate_input(grammar: str):
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, sample_path_for(grammar), grammar],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"{grammar}: rc={result.returncode}\n{result.stderr[-800:]}"
    assert int(result.stdout.strip()) > 0
