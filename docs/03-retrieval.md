# 3. Retrieval

Four channels, one router, and a decision about what to do when the answer is not in
the corpus. Every stage here has a number attached, and two of them ended up the
opposite of what was expected.

!!! done "What this stage owns"

    `search/routing.py` (the regex router), `search/retrieve.py` (channels, RRF,
    top-k, per-channel scores), `search/rerank.py` (the cross-encoder, off), and the
    three-band abstention thresholds in `config.py`.

## Routing: one regex, free MRR

An embedding has nothing useful to say about `handleAuthCallback`. It is not a word, it
carries no distributional meaning, and the nearest vectors to it are other camelCase
identifiers that happen to look similar. BM25 finds it exactly, because that is what
lexical search is for.

For a plain sentence it is the other way round.

```python
# search/routing.py
_SYMBOL_SHAPED = re.compile(r"^[A-Za-z_][\w.\-/:]*$")
_CAMEL_BOUNDARY = re.compile(r"[a-z0-9][A-Z]")

def looks_like_symbol(query: str) -> bool:
    """A single token with a boundary in it: camelCase, snake_case, kebab-case, a.b.c, a/b.

    Deliberately narrow: a false positive sends a real question to BM25 (measurably
    bad on prose); a false negative only gives up an improvement.
    """
```

The asymmetry in that docstring is the design. Getting it wrong in one direction costs
a good answer; getting it wrong in the other costs an improvement. So the pattern
requires a single token *and* an internal boundary — no spaces, and either camelCase, an
underscore, a dot, a slash, or all caps.

!!! measured "Why not always merge both channels"

    | Mode | Recall@8 | MRR | TR-prose R@8 | p50 |
    |---|---|---|---|---|
    | BM25 only | 0.405 | 0.240 | 0.263 | 2 ms |
    | dense only | 0.786 | 0.678 | 0.684 | 32 ms |
    | hybrid (always RRF) | 0.786 | 0.604 | 0.684 | 40 ms |
    | **auto** (symbol→BM25, prose→dense) ✓ | **0.786** | **0.690** | 0.684 | 34 ms |

    Always merging is *worse* than dense alone (MRR 0.604 vs 0.678), and the reason is
    mechanical: RRF promotes the best guess of the channel that failed. For a prose
    question, BM25's top result is a keyword coincidence, and merging gives it rank
    weight it did not earn.

    Routing costs one regex and buys 0.012 MRR over dense-only. An LLM router would buy
    the same thing for a network call.

## RRF, when it is used

`RAG_PROSE_MODE=hybrid` merges the two lists with Reciprocal Rank Fusion:
`Σ 1/(k + rank)`, `RAG_RRF_K=60`.

The reason it uses ranks and not scores is that the two scores are not on the same
scale — cosine lives in 0–1, BM25 in roughly 0–30. A weighted sum needs a normalization
constant, and that constant drifts with the corpus: the same weights that work on a
TypeScript monorepo are wrong on a Go service. Rank has no units.

## The cross-encoder that got turned off

The plan was ordinary: retrieve 40 candidates, have `bge-reranker-v2-m3` read each one
next to the question, keep the best 8. Cross-encoders reliably beat bi-encoders at
ordering, and the recall headroom is real — Recall@40 is 0.95 against Recall@8 of 0.786.

!!! measured "It made the ordering worse"

    | Setting | Recall@8 | MRR | TR-prose R@8 | p50 |
    |---|---|---|---|---|
    | auto + dense | 0.786 | **0.690** | 0.684 | 34 ms |
    | hybrid + rerank | 0.762 | 0.508 | 0.579 | 4389 ms |
    | auto + rerank | 0.762 | **0.514** | 0.579 | 2050 ms |

    MRR fell by a quarter and p50 went from 34 ms to seconds. `RAG_RERANK_ENABLED`
    defaults to `false`.

The +0.17 recall gap between k=8 and k=40 is still sitting there, unclosed. That is
stated rather than hidden, because it is the most obvious place for the next
improvement — and if a better reranker closes it, it enters as a new row in the ledger
rather than replacing this one.

This is what "measured" is for. The expected result was a gain; the observed result was
a loss; the flag exists so the next person can re-run the comparison on their corpus
instead of trusting either outcome.

## Every hit carries its channel scores

```json
{
  "path": "src/queue/worker.ts",
  "symbol": "enqueueWebhookEvent",
  "start_line": 42,
  "end_line": 71,
  "category": "code",
  "scores": {"dense": 0.71, "bm25": 12.4, "rrf": 0.031}
}
```

Which channel found what is in the data structure, not just in the UI. That is what
makes an ablation possible at all: the eval harness reads these to attribute a result
to a channel, and a debugging session reads them to explain why a wrong chunk came
first.

## Saying "no answer"

This is the part that separates a demo from something an agent can rely on.

A kNN search means "the nearest k". It has no concept of "nothing is near". Ask the
retriever *"have you ever been to a five-star resort?"* against a TypeScript API and it
returns eight chunks, confidently, with the best one at 0.366. Hand those to an LLM and
you have manufactured a hallucination out of an empty result.

The obvious fix — a cosine threshold — does not survive contact with the data:

!!! measured "Cosine does not separate in the grey zone"

    Golden top-1 dense: median 0.636, **min 0.526**.
    Irrelevant top-1 dense: median 0.531, **max 0.598**.

    The distributions overlap. Any single threshold either cuts real answers or admits
    junk.

So instead of a gate, three bands — CRAG's shape:

| Band | Setting | Behaviour |
|---|---|---|
| < 0.45 | `RAG_MIN_DENSE_SCORE` | **dropped**, counted in `dropped` |
| 0.45 – 0.55 | `RAG_WEAK_DENSE_SCORE` | returned, with `weak_match: true` |
| ≥ 0.55 | — | normal |

The floor sits well below the lowest real answer in the golden set, so it only clears
the absurd tail. The note is a **signal, not a filter** — the results still come back,
and the decision belongs to whoever consumes them: the agent, the LLM, or the human
looking at the UI.

!!! measured "Why the note and not a reranker gate"

    | Gate | abstain (13 negatives) | false_weak (42 positives) | added latency |
    |---|---|---|---|
    | dense < 0.55 note | 0.833 | 0.048 | 0 |
    | **floor 0.45 + note 0.55** ✓ | **0.846** | **0.048** | 0 |
    | rerank < 0.05 | 1.000 | **0.286** | +550 ms |
    | rerank < 0.5 | 1.000 | 0.595 | +550 ms |

    The reranker catches every negative — and gives a false alarm on **one query in
    three**. An agent that sees a warning it can safely ignore a third of the time
    learns to ignore it always. The cosine note fires wrongly 5% of the time, which is
    a signal worth keeping.

    The 0.45 floor dropped nothing from the golden set and took two negatives ("resort"
    at 0.366, "Kafka rebalance" at 0.418) to zero results.

### The thresholds are not universal

They were measured **for this embedding model and this corpus**. The cosine distribution
moves with the model — for the same job, Mistral's embeddings land around 0.73 and
Gemini's around 0.46. Shipping these two numbers as constants for someone else's stack
would be exactly the kind of unmeasured claim this project is trying to avoid.

So the eval report prints what you need to move them:

```json
"calibration": {
  "positive_min_top_dense": 0.498,
  "negative_max_top_dense": 0.587
}
```

Put the floor below the first number and the note between the two. Recalibrate whenever
the embedding model changes.

## Flags

Every one of these can be overridden per request in the `POST /search` body and in
`rag eval` parameters — the whole chain was built for ablation.

| Flag | Default | What it does |
|---|---|---|
| `RAG_SEARCH_MODE` | `auto` | symbol → BM25, prose → `RAG_PROSE_MODE` |
| `RAG_PROSE_MODE` | `dense` | `dense` or `hybrid` (dense + BM25 → RRF) |
| `RAG_RERANK_ENABLED` | `false` | 40 candidates → cross-encoder → 8 — measured, it hurt |
| `RAG_TOP_K` | `8` | results returned |
| `RAG_CANDIDATES` | `40` | pool handed to the reranker |
| `RAG_MIN_DENSE_SCORE` | `0.45` | hard floor; 0 turns it off |
| `RAG_WEAK_DENSE_SCORE` | `0.55` | the `weak_match` note threshold |

```bash
uv run rag search "handleAuthCallback" --mode bm25
uv run rag search "how are webhook events queued" -r my-api --json
```
