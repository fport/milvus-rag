# Running it on your own server

Two paths: **Docker Compose** (recommended — Milvus wants a container anyway) or keeping
Milvus in a container and running the service bare with **systemd**.

## Requirements

| Resource | Minimum | Note |
|---|---|---|
| OS | Linux x86_64 / arm64 | Docker 24+, compose v2 |
| RAM | 8 GB (16 GB is comfortable) | bge-m3 ~3-4 GB, Milvus ~2-3 GB |
| Disk | 20 GB+ | model cache ~5 GB + Milvus data + repo clones |
| CPU | 4+ cores | embedding on CPU ~5-8 chunks/s; a 3k-chunk repo ~8-10 min |
| GPU | not required | with one, `RAG_EMBEDDING_DEVICE=cuda` gives 5-10x |

Note: if the CPU feels slow, `RAG_EMBEDDING_BACKEND=openai` (text-embedding-3-large, needs
`OPENAI_API_KEY`) moves indexing to the API, and no model download is needed either.

## Path 1 — Docker Compose (recommended)

```bash
git clone <this-repo> && cd <repo>
cp .env.example .env        # AZURE_DEVOPS_* / GITHUB_TOKEN / RAG_WEBHOOK_SECRET / ANTHROPIC_API_KEY
docker compose -f infra/docker-compose.prod.yml up -d --build
docker compose -f infra/docker-compose.prod.yml logs -f rag   # wait for the "ready" line
curl -f http://localhost:8090/health
```

- On the first boot the embedding model (~2.2 GB) is downloaded from Hugging Face; it stays
  in the `hf-cache` volume, so later boots take seconds.
- The tree-sitter grammars (`PREFETCH_GRAMMARS`, ~60 languages) are downloaded during the
  image build and stay in the layer; no network is needed at runtime. If a language outside
  the list shows up, the pack tries to download it on first use, and with no network those
  files are indexed with plain windows (the log says "grammar could not be loaded").
- The `rag-data` volume holds SQLite (repo records, the manifest, jobs) and the repo clones.
  **That is the only thing to back up** — the data in Milvus is derived and can be
  regenerated with `rag sync --force`.
- Updating: `git pull && docker compose -f infra/docker-compose.prod.yml up -d --build`
  (if a chunk/embedding setting changed, the service does a full re-index automatically on
  the next sync).

## Path 2 — systemd (Milvus in a container, the service bare)

```bash
docker compose -f infra/docker-compose.yml up -d      # Milvus + etcd + MinIO only
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --no-dev
# download the grammars now (otherwise they arrive on first use; on an offline host it falls back to plain windows)
uv run python -c "from tree_sitter_language_pack import prefetch; from milvus_rag.sources.files import PREFETCH_GRAMMARS; prefetch(sorted(PREFETCH_GRAMMARS))"
```

`/etc/systemd/system/milvus-rag.service`:

```ini
[Unit]
Description=Milvus RAG
After=network-online.target docker.service

[Service]
User=rag
WorkingDirectory=/opt/milvus-rag
ExecStart=/home/rag/.local/bin/uv run rag serve
Restart=on-failure
EnvironmentFile=/opt/milvus-rag/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now milvus-rag
```

## Exposing it: reverse proxy + TLS

**The service has no authentication of its own.** Do not put 8090 straight on the internet:

- the webhooks have to be reachable from outside (Azure/GitHub will push to them),
- everything else (search, adding repos, reading files!) should stay on the internal network.

A practical split with Caddy (automatic TLS included) — `/etc/caddy/Caddyfile`:

```caddy
rag.example.com {
    # The webhook endpoints are public; they are secured by signature/secret anyway:
    # Azure → a shared secret, GitHub → HMAC-SHA256 (RAG_WEBHOOK_SECRET).
    handle /webhooks/* {
        reverse_proxy localhost:8090
    }
    # Everything else behind basic auth (generate the password: caddy hash-password)
    handle {
        basic_auth {
            admin <bcrypt-hash>
        }
        reverse_proxy localhost:8090
    }
}
```

With nginx the same split is `location /webhooks/ { proxy_pass ... }` plus `auth_basic` on
the rest. An alternative: never expose the service at all and use the poller instead of
webhooks (`RAG_POLL_INTERVAL_SECONDS`, 300 s by default) — the index refreshes at most 5
minutes after a push, and you open no ports.

## Webhook setup

- **Azure DevOps:** Project Settings → Service Hooks → Web Hooks → *Code pushed*
  → URL `https://rag.example.com/webhooks/azure/push`, Basic auth password (or the
  `X-RAG-Webhook-Secret` header) = `RAG_WEBHOOK_SECRET`.
- **GitHub:** repo → Settings → Webhooks → Add webhook → URL
  `https://rag.example.com/webhooks/github/push`, content type `application/json`,
  Secret = `RAG_WEBHOOK_SECRET`, "Just the push event".
  (GitHub's "Recent Deliveries" tab shows the deliveries and the responses; the service
  answers a ping event with `{"pong": true}`.)

## Connecting agents (MCP)

The service speaks MCP under `/mcp`. Locally:

```bash
claude mcp add --transport http milvus-rag http://localhost:8090/mcp
```

If it runs on a server, two things are needed: the endpoint has to be reachable (in the
reverse proxy `/mcp` can stay behind basic auth — an MCP client can send it with `--header`)
and **the host has to be allowed**:

```bash
RAG_MCP_ALLOWED_HOSTS=rag.example.com
```

That guard is for DNS rebinding; left empty, only MCP requests coming from localhost are
accepted (it does not affect the HTTP API).

## Health and operations

| What | How |
|---|---|
| Health | `GET /health` — Milvus connection, model, repo count |
| Job tracking | `GET /jobs?repo_id=...` — status + `stats.progress` |
| Logs | `docker compose ... logs -f rag` (one line, `key=value`) |
| Milvus UI | Attu with `--profile ui` (:8091) — keep it off in production |
| Full re-index | `POST /repos/{id}/sync {"force": true}` |

Known limits: a single worker (indexing is sequential — deliberate, see CLAUDE.md); Milvus
standalone is a single node. Both are enough for a one-server install.
