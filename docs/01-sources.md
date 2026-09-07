# 1. Sources and sync

Getting a repository in, and keeping it current, is the part of a code RAG that looks
trivial and is not. The retrieval quality of a stale index is zero no matter how good
the embeddings are.

!!! done "What this stage owns"

    `sources/` (Azure DevOps, GitHub, git, file discovery), `webhooks.py`, `jobs.py`,
    the SQLite tables in `db.py`, and the manifest diff at the top of
    `index/pipeline.py`.

## Three sources, one path

Azure DevOps and GitHub are reached over REST to list projects and repositories; the
clone itself is plain git. A local directory takes the same code path with the clone
step skipped — which is what makes trying the service out a one-liner.

```bash
uv run rag azure projects
uv run rag azure repos Platform
uv run rag add-azure Platform backend-api --branch develop

uv run rag github repos sindresorhus        # public repos need no token
uv run rag add-github sindresorhus/p-limit

uv run rag add-local ~/code/my-api --name my-api   # not even a git repo
```

### The PAT is never written anywhere

A personal access token in a remote URL ends up in `.git/config`, in `git remote -v`,
and in the text of the next error message. So it is not put there:

```python
# sources/git.py
"""The credential is never embedded in the URL and never written to the remote config:
it is passed to every command as `-c http.extraheader=AUTHORIZATION: Basic ...`."""

_SECRET = re.compile(r"(Basic|Bearer)\s+[A-Za-z0-9+/=_\-]+", re.IGNORECASE)
```

Error output is run through that pattern before it is logged or returned, because a
failing `git fetch` prints the command it tried.

## Freshness: a webhook with a poller behind it

Both providers are reduced to the same `PushEvent`, and each is verified differently
because each offers something different:

| Provider | Verification |
|---|---|
| Azure DevOps | a shared secret — Basic auth password, `X-RAG-Webhook-Secret`, or `?secret=` |
| GitHub | HMAC-SHA256 of the body (`X-Hub-Signature-256`) against the same secret |

A webhook alone is not a freshness guarantee: a missed delivery is a permanent gap,
and nothing in the system would ever notice. So the poller compares the upstream branch
head against the last indexed commit every `RAG_POLL_INTERVAL_SECONDS`. Cron alone
would mean a stale window; the webhook alone would mean silent gaps. The pair costs one
extra HTTP call per interval.

Only a push to the tracked branch opens a job, and a repeated `(repo, commit)` — Azure
retries — opens nothing.

## One worker, one pending job per repo

```python
# jobs.py
"""Why a single worker: indexing is CPU-bound (embedding) and writes to Milvus;
indexing two repos at once slows both down and speeds up neither."""
```

There is a one-pending-job-per-repo guarantee, the same idea as BullMQ's job-id dedupe:
if five pushes hit the same repo in a minute, one job runs and one waits. The other
three are not queued, because by the time the first finishes they would be indexing the
same commit.

Job records live in SQLite. After a restart, jobs left `running` become `failed` and
`queued` ones are re-queued — a crash mid-index is a visible failed job, not a job that
silently disappeared.

## Change detection by content hash

This is the decision that removes a whole category of bugs.

```python
# index/pipeline.py
"""Change detection is based on the content hash, not on a commit diff: there are no
git edge cases from renames, mode changes or submodules, and local (non-git)
directories take the same path."""
```

Every indexable file's sha256 is stored in the SQLite `files` table. A sync hashes the
working copy and compares:

| Outcome | What happens |
|---|---|
| added | chunk, embed, insert |
| changed | delete the chunks at that path, then insert the new ones |
| deleted | delete only |
| unchanged | **never embedded again** |

A `git diff` would be cheaper and would bring renames, mode changes, force-pushes and
submodule updates with it. Hashing every file costs under a second for five thousand
files, which is invisible next to embedding.

### Surviving an interruption

The ordering here is deliberate and worth stating, because the obvious ordering leaves
a hole:

```
remove the manifest row  →  delete the old chunks  →  write the new ones  →  put the row back
```

If the process dies anywhere in the middle, the file is *absent* from the manifest, so
the next sync treats it as new and re-indexes it. Had the row been updated first, a
crash between "update" and "write" would leave a file that the manifest calls current
and the index does not contain — a permanent, silent gap.

## What gets indexed

Not everything in a repository is worth embedding. `sources/files.py` filters by
directory, extension and size:

- `IGNORED_DIRS` — `node_modules`, `.git`, `dist`, build output, virtualenvs
- `IGNORED_SUFFIXES` — binaries, images, archives, lockfiles
- `MAX_JSON_BYTES = 64_000` — a large JSON file is data, not code

Each surviving file gets a `lang` and a `category` (code, document, config), which the
retriever later exposes as the `DOCUMENT` badge — so an agent can tell a code snippet
inside a design document apart from real code.

## After the write

Two things happen when a job finishes:

**The search cache is cleared.** Otherwise a query answered from cache keeps showing
line numbers from the previous commit — a failure that looks like a retrieval bug and
is not.

**`index_version` is checked.** If a `RAG_*` chunking or embedding setting changed, the
version changes with it and the next sync does a full re-index instead of an
incremental one. A chunk produced under different settings is not comparable to one
produced under the current ones, and mixing them silently degrades every result.

```bash
uv run rag sync my-api              # incremental
uv run rag sync my-api --force      # full re-index
uv run rag poll                     # check the upstream heads once
```
