# 5. Measurement

Every claim on the previous four pages is a row in a table somewhere. That is not a
stylistic choice — it is the only defence against the failure mode this kind of system
invites, which is shipping something that *feels* better.

!!! done "What this stage owns"

    `eval.py` (the golden-set runner), `evals/golden.example.jsonl` (the template),
    `evals/results/<date>_<tag>.json` (every run), and the Measurement ledger in the
    README.

## One rule

> Never ship what was not measured. Every retrieval change enters the table as a
> row, not as "it feels better."

Three of this project's decisions went **against** the intuition that motivated them:
the cross-encoder reranker was added and then turned off, the rerank-based abstention
gate was the best negative detector and still lost, and enrichment — the feature least
expected to matter — produced the largest single gain. None of those were visible by
reading code or by trying a handful of queries by hand.

## The golden set

A JSONL file, one question per line, labels written **by hand by reading the repo** —
never by running the system and accepting what it returns. Labelling from the system's
own output measures nothing; it just writes down what the system already does.

```json
{"q": "where is the JWT issued?",
 "expect": ["src/auth/session.ts::createSession"],
 "mode": "any", "kind": "prose-en"}
```

| Field | Meaning |
|---|---|
| `expect` | `path` or `path::symbol`. **Never line numbers** — lines shift when code changes, symbols do not |
| `mode` | `any` → one entry is enough; `all` (default) → every entry must come back |
| `kind` | a breakdown label (`prose-en` / `prose-tr` / `symbol` / `negative`) — the report splits by it |

The `kind` breakdown is what makes the table diagnostic rather than a single number.
An aggregate Recall@8 of 0.786 hides that symbol queries score 1.0 and Turkish prose
0.684 — and it is that split, not the average, that says which knob to turn.

### Negative cases are part of the set

```json
{"q": "have you ever been to a five-star resort?", "expect": []}
```

A kNN search means "the nearest k". It has no concept of *nothing is near*: for that
question the retriever still returns 8 chunks, the best at 0.366. That is precisely
where hallucination starts, and a golden set made only of answerable questions cannot
see it.

`abstain_rate` (how often the system said "no answer" on a negative) is reported next
to `false_weak_rate` (how often the same signal fires wrongly on a positive). Either
number alone is meaningless: a gate that abstains on everything scores a perfect
abstain rate.

!!! measured "Which abstention gate stayed"

    | Gate | abstain (negatives) | false_weak (42 positives) | added latency |
    |---|---|---|---|
    | dense < 0.55 note | 10/12 = 0.833 | 2/42 = 0.048 | 0 |
    | **floor 0.45 + note 0.55** ✓ | 11/13 = 0.846 | 2/42 = 0.048 | 0 |
    | rerank < 0.05 | 12/12 | 12/42 = 0.286 | +550 ms p50 |
    | rerank < 0.5 | 12/12 | 25/42 | +550 ms |

    The reranker catches every negative — and gives 0.039 to the p25 of the real
    answers. A false alarm on one query in three is enough for an agent to learn to
    ignore the note entirely, at which point a perfect detector detects nothing.
    The cosine note is worse at negatives and 5% wrong on positives, so it is the one
    that shipped.

## The runner

```bash
G=evals/golden.example.jsonl
uv run rag eval $G -r my-api --tag dense  --mode dense  --no-rerank
uv run rag eval $G -r my-api --tag bm25   --mode bm25   --no-rerank
uv run rag eval $G -r my-api --tag hybrid --mode hybrid --no-rerank
uv run rag eval $G -r my-api --tag rerank --mode auto   --rerank
```

Each run writes `evals/results/<date>_<tag>.json` with the report *and every per-case
row* — question, recall, reciprocal rank, latency, what came back, `top_dense`. Keeping
the rows is what lets a surprising aggregate be traced back to the two questions that
caused it, months later.

Repo-specific golden files are gitignored. Only the template is published: a golden set
is a list of your internal file paths.

## Calibration travels with the model

The thresholds in this project (`0.45` floor, `0.55` note) are not universal constants.
They were measured for **BGE-M3 on this corpus**. The same job on Mistral's embeddings
lands around 0.73, on Gemini's around 0.46. Copying a number across models produces a
gate that either passes everything or blocks everything.

So the report carries a `calibration` block:

```json
"calibration": {"positive_min_top_dense": 0.498, "negative_max_top_dense": 0.587}
```

The lowest top-dense among positives that were found, and the highest among negatives.
The recipe is mechanical: **floor below the first, note between the two.** Change the
embedding model, re-run the eval, read the block, move the two numbers.

The overlap in those two values is also the honest reason the gate is a *note* and not
a decision. Positives bottom out at 0.498 while negatives reach 0.587 — the
distributions cross. No single cosine threshold separates them, so the system reports
what it sees and the consumer decides.

## The ledger

The full table lives in the README, kept in both languages, dated, with the corpus and
the question mix stated. Reproduced here in brief:

!!! measured "Retrieval modes — 318-file TypeScript monorepo, 42 questions, k=8"

    | Tag | Setting | Recall@8 | MRR | TR-prose R@8 | p50 |
    |---|---|---|---|---|---|
    | bm25 | BM25 only | 0.405 | 0.240 | 0.263 | 2 ms |
    | hybrid | dense+BM25 → RRF | 0.786 | 0.604 | 0.684 | 40 ms |
    | dense | dense only | 0.786 | 0.678 | 0.684 | 32 ms |
    | **auto+dense** ✓ | symbol→bm25, prose→dense | **0.786** | **0.690** | 0.684 | 34 ms |
    | hybrid k=40 | candidate pool | 0.952 | — | 0.895 | 39 ms |
    | hybrid+rerank | 40→8, bge-reranker-v2-m3 | 0.762 | 0.508 | 0.579 | 4389 ms |

    Recall@40 is 0.952. There is a real +0.17 sitting in the candidate pool that a
    reranker could close — and the one that was tried broke the ordering instead and
    cost seconds. The gap stays on the record as an open item, not as a fixed defect.

!!! measured "Chunking — AST vs. plain windows, same corpus indexed twice"

    | Tag | chunks | Recall@8 | MRR | TR-prose MRR | symbol MRR |
    |---|---|---|---|---|---|
    | **ast** ✓ | 2761 | **0.786** | **0.690** | 0.570 | **1.000** |
    | plain | 2100 | 0.762 | 0.598 | 0.518 | 0.321 |

    Recall moves by one question. MRR moves by 0.09, and almost all of it is symbol
    queries. AST chunking does not find more — it puts the right piece on top and knows
    its name. Which is also the size of what a language without a grammar loses:
    ranking and citation, not recall.

!!! measured "Enrichment — 46 files / 423 chunks, indexed twice"

    | Tag | Recall@8 | MRR | TR-prose R@8 / MRR | false_weak |
    |---|---|---|---|---|
    | subset-plain | 0.929 | 0.839 | 0.895 / 0.778 | 0.119 |
    | **subset-enriched** | **1.000** | **0.912** | **1.000 / 0.932** | **0.048** |

    The largest single gain in the project, from the feature that ships **off**. +0.15
    MRR on non-English prose, `false_weak` halved, abstention unmoved — the descriptions
    raised confidence on real answers without manufacturing it for irrelevant ones.
    It stays off because the default install has no LLM budget: ~11 descriptions/min on
    a local 9B means ~4 hours for a 2.8k-chunk repo. It is incremental after that,
    cached by chunk hash + model.

## What the ledger is for

Three things, none of them decoration:

1. **It makes reversal cheap.** The reranker is still in the codebase behind
   `RAG_RERANK_ENABLED`, with the row that says why it is off. A better reranker enters
   as a new row against the same 42 questions — not as an argument.
2. **It sets the size of expectations.** "AST chunking is worth +0.09 MRR, mostly on
   symbols" is a usable statement. "AST chunking is better" is not.
3. **It records the surprises.** Enrichment was the throwaway feature and it won.
   That only became knowable because it was measured before it was judged.
