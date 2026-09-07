# CLAUDE.md — milvus-rag

A RAG service over a codebase. You pick a repo from Azure DevOps, GitHub (or a local
directory); it is split into code units with tree-sitter, written to Milvus with
BGE-M3 + BM25, optionally re-ranked with a cross-encoder; when the repo changes, only the
changed files are re-indexed.

A standalone Python 3.12 + uv package; it talks to the outside world over HTTP only.
The repos it indexes are read-only — the clones are never written to.

## Commands

```bash
uv sync                                              # environment (uv downloads Python 3.12)
docker compose -f infra/docker-compose.yml up -d     # Milvus + etcd + MinIO (:19530)
cp .env.example .env                                 # fill in AZURE_DEVOPS_*, RAG_WEBHOOK_SECRET

uv run rag serve                                     # FastAPI :8090  (docs: /docs)
uv run rag azure projects | uv run rag azure repos <project>
uv run rag add-azure <project> <repo> [--branch main]  # register + index
uv run rag add-github <owner/repo> [--branch main]   # no token needed for public repos
uv run rag add-local ~/code/my-api --name my-api     # local directory (for trying it out)
uv run rag sync <repo-id> [--force]                  # incremental / full re-index
uv run rag search "..." [-r <repo-id>] [--mode bm25] [--no-rerank] [--json]
uv run rag ask "..."                                 # cited answer (needs an LLM)
uv run rag eval evals/golden.example.jsonl -r my-api --tag <tag>
uv run pytest -q && uv run ruff check src tests      # tests + lint
```

## Architecture

```
sources/   azure.py + github.py (REST: repo/ref) · git.py (clone/fetch, PAT via header) · files.py (file filters, sha256)
index/     chunk.py (tree-sitter AST chunker + markdown + fallback) · scrub.py (secrets/PII)
           embed.py (BGE-M3 | OpenAI) · store.py (Milvus: dense + BM25 sparse, repo_id partition key)
           enrich.py (optional LLM description, cached) · pipeline.py (manifest diff → chunk → embed → write)
search/    routing.py (symbol → BM25) · retrieve.py (channels → RRF → rerank, channel scores) · rerank.py · answer.py
db.py      SQLite: repos, files (path→sha), jobs, webhook_events, enrichment
jobs.py    single-worker queue + per-repo dedupe + Azure poller
webhooks.py Azure "Code pushed" + GitHub push (HMAC) · api.py FastAPI (+ /mcp mount) · cli.py typer
mcp_server.py MCP server (streamable HTTP): search_code · read_code · list_repos — signals (weak match, DOCUMENT, freshness) · eval.py golden runner (+ negative cases)
```

Data flow: `push → webhook/poll → job → refresh_source (fetch+reset) → compare file shas
against the manifest → delete + rewrite the changed file's chunks → update the manifest →
clear the retriever cache`.

## Binding decisions

- **Change detection by content hash, not by commit diff.** No git edge cases from renames,
  mode changes or submodules; local directories take the same path. The manifest row is
  removed before the chunks are deleted and put back after they are written — an
  interrupted job never leaves a gap.
- **A chunk is a code unit, ≤ 2000 bytes.** Large classes are split into members, small ones
  are merged, and comments/imports stick to the unit that follows. `text` stays clean; the
  header (file, symbol, class, imports) goes only into `indexed_text`.
- **Language coverage is a rule, not a table.** If the extension name is a grammar name in
  `tree-sitter-language-pack` (`.lua`, `.vue`, `.razor`, `.zig`), that grammar is used; the
  ones whose names differ (`.ts`, `.cs`, `.kt`) come from a small alias table validated
  against the pack (`files.py`). Anything without a grammar is still indexed (plain
  windows) — measured: AST adds one question in recall and +0.09 MRR (README → "Chunk
  ablation"), which is why we do NOT write rules per language/framework; there are no role
  labels and no regex symbol extractors. A grammar that crashes or hangs goes into
  `CODE_WITHOUT_GRAMMAR` (today `sql`, `cobol`); a new grammar enters via
  `PREFETCH_GRAMMARS` + `RAG_LIVE=1 pytest tests/test_grammars_live.py`. Pack ≥ 1.15
  downloads a grammar on first use → the Dockerfile bakes them in with `prefetch()`.
- **Symbol-shaped query → BM25, plain sentence → dense; rerank off.** Measured (README →
  Measurement ledger): auto+dense 0.786/0.690, symbols 1.0/1.0 on BM25;
  bge-reranker-v2-m3 hurt (0.762/0.508, p50 2-4 s) → off by default. Every hit carries its
  channel scores (`dense`, `bm25`, `rrf`, `rerank`).
- **Every retrieval change is measured with `rag eval`.** "It feels better" is not a result;
  no flag default changes until a JSON lands under `evals/results/`.
- **Enrichment (LLM descriptions) starts off.** BGE-M3 is multilingual; turn it on if you
  need it for non-English questions, but measure first. Descriptions are cached by chunk
  hash, and the wrong alphabet is rejected. The output language is `RAG_ENRICH_LANGUAGE`
  (default English) — the measured gain comes from adding prose in the language people ask
  in, so set it to match your question traffic.
- **One Milvus collection, `repo_id` as the partition key.** Changing the schema means
  rebuilding the collection. When a `RAG_*` chunk/embedding setting changes, `index_version`
  changes and the next sync does a full re-index.
- **MCP in the same process, over the same retriever.** Streamable HTTP under `/mcp`;
  blocking work (embedding, Milvus, files) runs through `anyio.to_thread` so the MCP session
  keeps flowing on a single event loop. Tool output is marked to the agent as DATA, not as
  instructions.
- **Irrelevance is handled in three bands (CRAG).** kNN never says "nothing is close", and
  cosine does not separate in the grey zone (measured, README → Abstention); a reranker gate
  gives 29% false alarms. So: dense < 0.45 is dropped (`min_dense_score`, counted as
  `dropped` — the absurd tail), 0.45-0.55 comes back with a `weak_match` note, above that is
  normal. Alongside it: the DOCUMENT label, index freshness, and a `read_code` locked to the
  manifest; the adjudicator is the agent/LLM/human. The golden set has negative cases
  (`expect: []`); `abstain` and `false_weak` are read together.
- **LLM provider `auto`.** Claude if an Anthropic key exists, otherwise OpenAI, otherwise
  local Ollama (`qwen3.5:9b`; measured, README → Local LLM). Thinking models get
  `think: false` and `num_ctx` 16k. The OpenAI client takes a `base_url`: vLLM/LM Studio
  connect the same way.
- **The PAT is never written anywhere.** It is passed to git with `-c http.extraheader=`;
  error messages are redacted. `scrub` runs before anything reaches the index.

## Rules

- English comments and identifiers. A comment explains the "why".
- Anything crossing a module boundary is typed in `models.py`; no bare dicts travel.
- Every new module has a counterpart under `tests/`. Tests that need Milvus or a model are
  marked `live`; unit tests run against a fake embedder/store.
- Anthropic calls go through `llm.py`, `claude-opus-5`, streaming + `get_final_message`,
  `fallbacks="default"`. Do not open an `anthropic.Anthropic()` anywhere else.
- Write measured numbers into the "Measurement ledger" in README.md (and mirror them into
  README.tr.md); the output of a command that actually ran, not an estimate.
- The documentation site lives in `docs/` (MkDocs Material, published to
  https://fport.github.io/milvus-rag/). Every page is bilingual: `<page>.md` is English and
  `<page>.tr.md` its Turkish counterpart — change one, change the other. A number that goes
  into the README ledger and is also quoted on a docs page has to move in both places.
  `uv run mkdocs serve` to preview, `uv run mkdocs build --strict` to check the links.
