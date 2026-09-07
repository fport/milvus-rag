# Rung 3 — Lexical search and routing

An embedding answers "what is this about". A great many real queries are not about
anything — they are a name.

```text
handleAuthCallback        ERR_CONN_RESET        RAG_WEAK_DENSE_SCORE
UserNotFoundException     invoice_line_items    --no-verify
```

For those, similarity is the wrong question. You do not want the nearest thing, you want
*that* thing, and the standard tool for it is thirty years older than the embedding:
**BM25**, exact term matching weighted by rarity.

## Two channels

Add a lexical index next to the vector one. Every chunk goes into both.

| Channel | Good at | Blind to |
|---|---|---|
| dense (embeddings) | paraphrase, cross-language, "what is this about" | exact identity, rare tokens, misspelt-but-exact |
| BM25 (lexical) | identifiers, error codes, config keys, quoted strings | synonyms, another language, anything reworded |

In this project both live in **one Milvus collection**: nobody writes the sparse field,
Milvus' own BM25 Function derives it from `indexed_text`. That removes the usual cost of
this rung — two indexes that must be kept consistent — and is most of why Milvus is
here. If your store cannot do that, this rung means running a second index and writing
to both, transactionally-ish.

## Then: do not merge them by default

This is the part that surprises people, and it is measured.

The obvious design is "always run both, fuse the results". Reciprocal Rank Fusion is the
usual fuser: `Σ 1/(k + rank)`, ranks rather than scores because cosine lives in 0–1 and
BM25 in roughly 0–30, and any weighted sum of the two needs a normalization constant
that drifts with the corpus.

!!! measured "Always-hybrid is worse than dense alone"

    | Mode | Recall@8 | MRR | non-English R@8 | p50 |
    |---|---|---|---|---|
    | BM25 only | 0.405 | 0.240 | 0.263 | 2 ms |
    | dense only | 0.786 | 0.678 | 0.684 | 32 ms |
    | hybrid (always RRF) | 0.786 | **0.604** | 0.684 | 40 ms |
    | **auto** (route by shape) ✓ | **0.786** | **0.690** | 0.684 | 34 ms |

    Fusing always **cost 0.074 MRR** against just using the dense channel.

The mechanism is not subtle once you see it: for a prose question, BM25's top result is
a keyword coincidence. RRF does not know that. It promotes the best guess of the channel
that failed, and that guess displaces a real answer.

Fusion assumes both channels had a fair shot at the query. For a query only one of them
can serve, it is actively harmful.

## Route instead

Decide which channel the query belongs to, and use that one.

```python
_SYMBOL_SHAPED = re.compile(r"^[A-Za-z_][\w.\-/:]*$")
_CAMEL_BOUNDARY = re.compile(r"[a-z0-9][A-Z]")

def looks_like_symbol(query: str) -> bool:
    """A single token with a boundary in it: camelCase, snake_case, kebab-case, a.b.c, a/b.

    Deliberately narrow: a false positive sends a real question to BM25 (measurably
    bad on prose); a false negative only gives up an improvement.
    """
```

That docstring is the design. The two errors are not symmetric:

- **false positive** (a real question routed to BM25) → a bad answer
- **false negative** (a symbol routed to dense) → a missed improvement

So the pattern is deliberately narrow: one token, no spaces, and an internal boundary.
Everything else is prose.

A regex, and it buys 0.012 MRR over dense-only while making symbol queries perfect
(BM25 scores 1.0/1.0 on them). An LLM router would buy the same thing for a network
call and a source of non-determinism.

!!! done "When hybrid *is* the right default"

    Routing works here because code queries fall cleanly into two shapes. On a corpus
    where they do not — mixed natural-language queries that also contain product names,
    SKUs or part numbers — there is no clean rule, and always-RRF is the honest default.

    The rung is "have both channels and use them deliberately". `RAG_PROSE_MODE=hybrid`
    turns fusion back on here; measure it on your own corpus rather than inheriting this
    table.

## What it does not fix

The retriever now finds the right thing for both query shapes. It will also, with equal
confidence, return eight chunks for a question about a five-star resort.

That is **[rung 4](04-honesty.md)**.
