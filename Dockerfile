# One image: the FastAPI service + the index worker (same process).
#
#   docker compose -f infra/docker-compose.prod.yml up -d --build
#
# The model is NOT baked into the image (bge-m3 ~2.2 GB): it is downloaded on the first
# boot and stays in the hf-cache volume — the image stays small, and updating the model
# does not change the image. Thanks to the pytorch-cpu source in pyproject, torch comes
# without CUDA on Linux.

FROM python:3.12-slim

# git: required to clone repos. curl: used by the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    RAG_DATA_DIR=/app/data \
    HF_HOME=/app/.cache/huggingface

# Dependencies first: this layer is only rebuilt when the lock file changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# tree-sitter-language-pack >= 1.15 downloads grammars on first use. So that the first
# .kt / .php file on an offline host does not stall on a download and fall back to plain
# windows, the smoke-tested list (files.py -> PREFETCH_GRAMMARS) is baked into the image.
# It writes under ~/.cache; that is a layer, not a volume. The service runs as the same
# user (root).
RUN uv run python -c "from tree_sitter_language_pack import prefetch; \
    from milvus_rag.sources.files import PREFETCH_GRAMMARS; prefetch(sorted(PREFETCH_GRAMMARS))"

EXPOSE 8090
CMD ["uv", "run", "rag", "serve"]
