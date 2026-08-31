import os
from pathlib import Path

import pytest

# Testler .env okumasın; ortamdan gelen gerçek anahtarlar sonucu değiştirmesin.
os.environ.setdefault("RAG_DATA_DIR", str(Path(__file__).parent / ".data"))


@pytest.fixture
def tmp_settings(tmp_path: Path):
    from milvus_rag.config import Settings

    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        rerank_enabled=False,
        cache_ttl_seconds=0,
        webhook_secret="s3cret",
    )


@pytest.fixture
def db(tmp_settings):
    from milvus_rag.db import Database

    return Database(tmp_settings.db_path)
