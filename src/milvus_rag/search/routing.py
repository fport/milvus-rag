"""Choosing the retriever from the shape of the query.

Measured in an earlier RAG experiment (30 questions):

| recall@5      | symbol search  | prose     |
|---------------|----------------|-----------|
| dense         | 0.60           | 0.62      |
| BM25          | 0.80           | 0.36      |
| RRF (both)    | 0.60           | 0.52      |

For a symbol-shaped query (`handleAuthCallback`, `QUEUE_NAMES`, `auth.service`) an
embedding has nothing to say; BM25 does. For a plain sentence it is the other way
round. Always merging the two promotes the best guess of the channel that failed
to find it, at full strength. Routing costs one regex.
"""

import re

_SYMBOL_SHAPED = re.compile(r"^[A-Za-z_][\w.\-/:]*$")
_CAMEL_BOUNDARY = re.compile(r"[a-z0-9][A-Z]")


def looks_like_symbol(query: str) -> bool:
    """A single token with a boundary in it: camelCase, snake_case, kebab-case, a.b.c, a/b.

    Deliberately narrow: a false positive sends a real question to BM25 (measurably
    bad on prose); a false negative only gives up an improvement.
    """
    text = query.strip()
    if not text or any(character.isspace() for character in text):
        return False
    if not _SYMBOL_SHAPED.match(text):
        return False
    return bool(
        _CAMEL_BOUNDARY.search(text)
        or "_" in text
        or "-" in text
        or "." in text
        or "/" in text
        or "::" in text
        or (text.isupper() and len(text) > 1)
    )
