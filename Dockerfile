# Tek imaj: FastAPI servisi + index worker'ı (aynı süreç).
#
#   docker compose -f infra/docker-compose.prod.yml up -d --build
#
# Model imaja GÖMÜLMEZ (bge-m3 ~2.2 GB): ilk açılışta iner ve hf-cache
# volume'ünde kalır — imaj küçük kalır, model güncellemesi imaj değiştirmez.
# torch, pyproject'teki pytorch-cpu kaynağı sayesinde Linux'ta CUDA'sız gelir.

FROM python:3.12-slim

# git: repo klonlamak için zorunlu. curl: compose healthcheck kullanıyor.
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

# Önce bağımlılıklar: bu katman yalnızca lock değişince yeniden kurulur.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8090
CMD ["uv", "run", "rag", "serve"]
