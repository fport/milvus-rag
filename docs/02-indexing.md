# 2. Indexing

Turning a repository into something searchable. Four steps, and the first one decides
how good the other three can be.

!!! done "What this stage owns"

    `index/chunk.py` (tree-sitter), `index/scrub.py` (secrets and PII),
    `index/embed.py` (BGE-M3), `index/enrich.py` (optional descriptions), and
    `index/store.py` (one Milvus collection).

## A chunk is a code unit

The default chunker in most RAG stacks is a character window. On prose that is
defensible. On code it cuts a function in half, and half a function is not a thing
anyone can answer a question with.

So the boundaries come from a real parser — tree-sitter, not brace counting, because a
`}` inside a string literal defeats brace counting on the first file that has one.

Four rules:

1. **A large class or namespace is split into its members.** The header and fields
   become their own chunk.
2. **Small things are merged with a neighbour.** A one-line type, a short const, an
   import block, a JSDoc comment — nothing goes below `RAG_CHUNK_MIN_BYTES`.
3. **No chunk exceeds `RAG_CHUNK_MAX_BYTES`** (2000 B, roughly 500 tokens). BGE-M3
   accepts 8192 tokens, and that is exactly the trap: the embedding of a long chunk
   becomes an *average* and represents nothing well. A function that exceeds the ceiling
   on its own is split into line windows that keep its symbol name.
4. **`text` stays clean.** It is what reaches the LLM and the citation. The "what am I,
   where do I live" header — file, symbol, parent class, imports — goes only into
   `indexed_text`, the field that is embedded and BM25-indexed.

Rule 4 is the one that is easy to get wrong. Putting the header into `text` makes every
citation start with three lines of metadata; leaving it out of `indexed_text` makes a
method called `handle` indistinguishable from forty others.

### Language coverage is a rule, not a table

```python
# The extension name is the grammar name for most languages: .lua, .vue, .razor, .zig.
# The ones that differ (.ts, .cs, .kt) come from a small alias table validated against
# the pack. Anything without a grammar is still indexed, as plain windows.
CODE_WITHOUT_GRAMMAR: dict[str, str] = {".sql": "sql", ".cob": "cobol", ".cbl": "cobol"}
```

`tree-sitter-language-pack` ships 371 grammars. Writing a per-language rule table would
mean maintaining 371 entries and being wrong about the 372nd; deriving the grammar from
the extension means a new language works the day the pack supports it.

Two grammars are excluded by name because they crash or hang the parser — `sql` and
`cobol`. They are still indexed, with line windows.

!!! measured "What AST chunking is actually worth"

    Same corpus indexed twice, once with tree-sitter and once with plain windows,
    same golden set, k=8:

    | Chunker | chunks | Recall@8 | MRR | symbol R@8 / MRR | TR-prose MRR |
    |---|---|---|---|---|---|
    | **AST** ✓ | 2761 | **0.786** | **0.690** | 1.0 / **1.0** | 0.570 |
    | plain windows | 2100 | 0.762 | 0.598 | 0.75 / 0.321 | 0.518 |

    Recall differs by one question. MRR differs by 0.09, and **almost all of it is in
    symbol queries** (MRR 0.321 → 1.0). On English prose the plain window ties and is
    even one question ahead.

    AST chunking does not find more. It puts the right piece on top and says its name.
    For a language left without a grammar the loss is ranking and citation quality, not
    recall — which is why a missing grammar is not an emergency.

## Redaction before storage

A vector cannot be reversed. The `content` field stored next to it can — it is returned
verbatim on every citation, and from there it travels into the cache, the logs and the
model's answers. So scrubbing happens before anything reaches the index.

The rules are deliberately conservative, and there is a number behind that word:

!!! measured "Conservative beats thorough"

    On a real repository, a rule that redacted based on *names* produced **247
    redactions across 78 files**, and almost none of them were secrets: `token: text(`
    is a database column, `secret: string` is a type annotation.

    The shipped rules made **2 redactions** on the same repository. One of them was a
    real token.

The patterns match shapes, not names — `sk-`, `ghp_`, `xox[baprs]-`, `AKIA…`, JWTs,
emails, phone numbers — plus one name-based rule that only fires on a SCREAMING_SNAKE
key with an assigned value:

```python
# DB_PASSWORD=hunter2 is a secret; `promptTokens` and `this.accessToken` are not.
_ASSIGNED_SECRET = re.compile(
    r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]*(?:PASSWORD|SECRET|TOKEN|APIKEY|API_KEY|PAT)[A-Z0-9_]*"
    r"|(?:PASSWORD|SECRET|TOKEN|APIKEY|API_KEY|PRIVATE_KEY)[A-Z0-9_]*)"
    r"(\s*[:=]\s*)"
    r"(\"[^\"\n]*\"|\'[^\'\n]*\'|[^\s\"\'#,;]+)"
)
```

…and an exclusion list for the things that look like assignments and are not:
`process.env.X`, `${…}`, a type name, a function call, `this.…`, a URL scheme, and the
placeholder vocabulary (`your-`, `change-me`, `xxx`, `example`).

Reserved email domains (`example.com`, `*.test`, `*.invalid`, `localhost`) are left
alone, because redacting the example address in a doc comment removes information and
protects nobody.

## Embedding

BGE-M3, 1024 dimensions, running locally by default. Two properties matter.

**It is multilingual**, which is the entire reason a question in one language can find
code commented in another. The earlier experiment with MiniLM is the control group:
Turkish prose scored **Recall@5 = 0.04**. The same questions against BGE-M3 score
0.684. That is not a tuning difference, it is a different capability.

**It runs on the machine**, so the code never leaves. `RAG_EMBEDDING_BACKEND=openai`
switches to a hosted model for anyone who prefers the tradeoff.

```python
# index/embed.py
"""Vectors are L2-normalized on the way out and searched with COSINE — mixing metrics
silently produces wrong ordering, so it is done in one place."""
```

Normalization in one place is not fussiness. A normalized vector searched with inner
product and an unnormalized one searched with cosine give *plausible* orderings that
are subtly wrong, and nothing errors.

The model is loaded once per process, not per request — the first load is 10–20 seconds.
Torch is imported inside the functions, so `import milvus_rag` in the CLI does not pull
in a deep-learning stack to print a help message.

## Enrichment, off by default

Code contains almost no natural language, and what it does contain is in whatever
language its author wrote comments in. The cure is not a better matcher; it is writing
the missing prose. This is Anthropic's contextual retrieval idea: give the LLM the whole
file and ask it for two or three sentences per chunk, then embed those alongside.

!!! measured "What enrichment buys, and what it costs"

    46 files / 423 chunks, the same corpus indexed twice, descriptions from a local
    `qwen3.5:9b`:

    | Arm | Recall@8 | MRR | TR-prose R@8 / MRR | false_weak |
    |---|---|---|---|---|
    | plain | 0.929 | 0.839 | 0.895 / 0.778 | 0.119 |
    | **enriched** | **1.000** | **0.912** | **1.000 / 0.932** | **0.048** |

    The gain lands exactly where it was predicted: **+0.15 MRR on Turkish prose**,
    +0.01 on English. `false_weak` halved, because the dense scores of real answers rise
    and the 0.55 note fires wrongly less often. Abstention on negatives did not move —
    the descriptions did not manufacture confidence for an irrelevant question.

    Cost: ~11 descriptions/minute on a local 9B, so a 2.8k-chunk repo takes about four
    hours the first time. It is incremental after that (cached by chunk hash + model),
    and minutes with a cloud model.

It stays **off** by default because on a keyless install every `add-*` would spin Ollama
for hours. Turn it on when your question traffic is not in English, and set
`RAG_ENRICH_LANGUAGE` to match — the measured gain comes from prose in the language
people actually ask in.

Two guards, both from watching a 7B model fail:

- **A coverage check.** A small model will happily return one object and drop the rest
  in silence. The response is checked against the chunk list.
- **An alphabet check.** `qwen2.5:7b` slides into Chinese mid-sentence under load and
  does not mention it. A description in the wrong script is rejected.

## One collection, `repo_id` as the partition key

```python
# index/store.py
"""Dense and sparse live in the same collection. Nobody writes the sparse field;
Milvus' own BM25 Function derives it from `indexed_text`, so there is no separate
lexical index."""
```

That last clause is why Milvus was chosen over the alternatives. Hybrid search normally
means maintaining two indexes and keeping them consistent; here BM25 is a function over
a field in the same collection, so a write is a write.

The partition key means a `repo_id in [...]` filter descends only into the relevant
partitions rather than scanning and filtering.

Deletes and updates go through `repo_id + path`. There is no archiving and no soft
delete: what is in Milvus is derived data, and the source of truth is the repository.
Anything that cannot be rebuilt from the repo does not belong in there.

```bash
# The whole indexing path, from a directory to a searchable collection
uv run rag add-local ~/code/my-api --name my-api
uv run rag sync my-api --force     # re-chunk and re-embed everything
```
