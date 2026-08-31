"""Tek satır, okunur log. api/ tarafındaki pino çıktısına benzer: zaman, seviye,
bağlam, mesaj ve anahtar=değer alanları."""

import logging
import sys
from typing import Any


class _KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = getattr(record, "kv", None)
        if not extras:
            return base
        rendered = " ".join(f"{key}={_short(value)}" for key, value in extras.items())
        return f"{base} {rendered}"


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 200 else text[:197] + "..."


class Logger(logging.LoggerAdapter):  # type: ignore[type-arg]
    """`log.info("mesaj", repo=..., files=...)` biçimini destekler."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        reserved = {"exc_info", "stack_info", "stacklevel", "extra"}
        kv = {key: value for key, value in kwargs.items() if key not in reserved}
        for key in kv:
            kwargs.pop(key)
        extra = kwargs.get("extra") or {}
        extra["kv"] = {**(extra.get("kv") or {}), **kv}
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        _KeyValueFormatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Gürültülü üçüncü partiler.
    for noisy in ("httpx", "httpcore", "sentence_transformers", "pymilvus", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> Logger:
    return Logger(logging.getLogger(f"rag.{name}"), {})
