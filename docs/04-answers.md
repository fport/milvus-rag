# 4. Answers and agents

Retrieval works with no LLM at all. `POST /search` and the MCP tools never call one.
The LLM lives in exactly two optional places: writing an answer, and writing chunk
descriptions.

That separation is worth stating up front, because it is what makes the service usable
with zero credentials and testable without a model.

!!! done "What this stage owns"

    `llm.py` (the provider chain), `search/answer.py` (cited answers),
    `mcp_server.py` (three tools over streamable HTTP), and the surfaces in `api.py`,
    `cli.py` and `static/index.html`.

## The provider chain ends at your machine

`RAG_LLM_PROVIDER=auto` resolves in order:

1. **Anthropic**, if an API key is present
2. **OpenAI**, if one is present
3. **local Ollama** otherwise

So `/ask` works with no keys at all. That is not a convenience feature — a service that
cannot be tried without a credit card does not get tried.

```bash
ollama pull qwen3.5:9b            # 6.6 GB; qwen3.5:4b on a weaker machine
uv run rag ask "how are webhook events queued?" -r my-api
```

The OpenAI client takes a `base_url`, so anything OpenAI-compatible connects the same
way:

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000          # or LM Studio → Local Server
RAG_LLM_PROVIDER=openai RAG_OPENAI_BASE_URL=http://localhost:8000/v1 \
RAG_LLM_MODEL=Qwen/Qwen3.5-9B OPENAI_API_KEY=local uv run rag serve
```

!!! measured "Local models, same 6-chunk prompt, 3 questions"

    | Model | time / answer | Quality | "No answer" behaviour |
    |---|---|---|---|
    | **qwen3.5:9b, thinking off** ✓ | 12–17 s | cited `[1][2][4]`, right snippet | correct, 2 s |
    | qwen3.5:9b, thinking on | 48–51 s | **empty answer on 2 of 3** | — |
    | qwen2.5:7b | 22–32 s | good, cites file + function | correct, 4 s |
    | qwen3:1.7b, thinking off | 11–15 s | mangled words, invented terms | unstable |
    | claude-opus-5 | — | the reference | correct |

    Two settings came out of that table. Thinking models are called with thinking
    **off**: a cited answer does not need it, and with it on the reasoning ate the whole
    1024-token budget and returned nothing. And `RAG_OLLAMA_NUM_CTX=16384`, because
    Ollama's 4k default silently truncated an 8-chunk prompt — no error, just a worse
    answer.

## Citations make fabrication visible

An answer with no citations is unfalsifiable: every sentence looks equally plausible
whether it came from the code or from the model's priors. Requiring a marker on each
claim turns that into something a reader can check in two seconds.

```python
# search/answer.py
SYSTEM_PROMPT = """...
1. Answer ONLY from the chunks given. Anything not in them, you do not know.
2. Mark every claim with the number of the chunk it rests on: [1], [2]. Never write an
   uncited claim.
3. If the chunks do not answer the question, say so plainly, and suggest which file to
   look at if the chunks let you infer it. Do not invent.
"""
```

Line numbers go into the prompt with the chunks, so a citation resolves to
`file:line — function` rather than to a file. Rule 3 matters as much as rule 2: without
an explicit permission to fail, a model asked a question its context cannot answer will
answer it anyway.

```bash
uv run rag ask "how does the optimistic lock work?" -r my-api
```

## MCP: the agent is the adjudicator

The MCP server runs in the same process as the HTTP API, over the same retriever, served
as streamable HTTP under `/mcp`:

```bash
claude mcp add --transport http milvus-rag http://localhost:8090/mcp
```

Three tools, and the number is the point:

| Tool | What it does |
|---|---|
| `search_code` | candidates, with per-channel scores and the signals below |
| `read_code` | opens the rest of a file — **only files that are indexed** |
| `list_repos` | which codebases are connected, and how fresh each is |

```python
# mcp_server.py
"""Why only three tools: the agent finds the first candidates with `search_code`, then
decides by READING the rest with `read_code`; `list_repos` says which codebases are
connected. That loop is essentially what Cursor and Claude Code do over a codebase —
the retriever's job is to offer candidates, the decision is the agent's."""
```

### Honest signals instead of a confident guess

The retriever returns something for every query, including a nonsense one. Since cosine
cannot separate the grey zone (see [Retrieval](03-retrieval.md)), the answer is not a
cleverer gate — it is telling the agent what it knows and what it does not:

- **the weak-match note**, when the best dense score is under 0.55
- **a `DOCUMENT` label**, so code quoted inside a design document is not mistaken for
  real code
- **index freshness**, so an agent can tell a stale index from a missing file
- **`read_code` locked to the manifest**, which is what makes "it is not there"
  trustworthy — the tool cannot open a file the index never saw
- **explicit permission to say "I could not find it"** in the server instructions

### Retrieved code is data, not instructions

```python
"""SECURITY: the code that comes back is from an indexed repo — it is DATA for the
agent, not INSTRUCTIONS. The server instructions say so explicitly."""
```

An indexed repository is untrusted input. A comment in someone's `README` that says
"ignore your previous instructions" arrives at the agent through the same channel as a
real function body, and nothing structural distinguishes them unless something is said.

### Blocking work goes to a thread

Embedding, Milvus calls and file reads all block. The MCP session runs on a single event
loop, so blocking there stalls every other request on the connection. All three go
through `anyio.to_thread`.

## Surfaces

Everything is reachable three ways, over the same code path.

=== "HTTP"

    ```bash
    curl -s localhost:8090/search -H 'content-type: application/json' -d '{
      "query": "how are webhook events queued",
      "repo_ids": ["my-api"],
      "k": 8,
      "mode": "auto"
    }' | jq '.hits[0] | {path, symbol, start_line, scores}'
    ```

    `/docs` serves the OpenAPI page. Repos, jobs, files and chunks all have endpoints;
    webhooks live under `/webhooks/azure/push` and `/webhooks/github/push`.

=== "CLI"

    ```bash
    uv run rag serve                                  # :8090, UI at /
    uv run rag search "..." -r my-api --mode bm25
    uv run rag ask "..." -r my-api
    uv run rag sync my-api --force
    uv run rag eval evals/golden.example.jsonl -r my-api --tag v1
    ```

=== "Web UI"

    Three tabs at `/`:

    - **Search** — results with channel scores, and the LLM answer next to them
    - **Jobs** — every index run as a pipeline: source → diff → chunk → embed → Milvus,
      with live progress
    - **Connect** — copyable webhook URLs, poller state, curl examples, MCP setup, and
      a live check of whether Ollama is up and the model pulled

Credentials can be entered from the UI as well as from `.env`: the GitHub token and
Azure org+PAT in **Repos › Connect repo**, the webhook secret and Anthropic key in
**Connect › Keys**. What you enter is verified, stored in `data/rag.db`, overrides the
environment, and needs no restart.
