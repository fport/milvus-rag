"""Push webhook'ları: Azure DevOps "Code pushed" ve GitHub "push".

İki sağlayıcı da aynı `PushEvent`'e indirgenir; API katmanı olayı sağlayıcıya
göre repo kaydıyla eşleştirir. Yalnızca izlenen branch'e gelen push iş açar;
aynı (repo, commit) ikinci kez gelirse (yeniden deneme) iş açılmaz.

Doğrulama sağlayıcıya göre:
- Azure: paylaşılan sır — Basic auth şifresi, `X-RAG-Webhook-Secret`
  header'ı ya da `?secret=`.
- GitHub: gövdenin HMAC-SHA256 imzası (`X-Hub-Signature-256`), sır aynı
  RAG_WEBHOOK_SECRET.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PushEvent:
    external_id: str  # Azure repo GUID'i ya da GitHub sayısal id'si
    repo_name: str
    project: str  # Azure projesi ya da GitHub owner'ı
    branch: str
    old_commit: str
    new_commit: str


# ------------------------------------------------------------------- azure


def parse_azure_push(payload: dict[str, Any]) -> list[PushEvent]:
    """Bir push birden fazla ref güncelleyebilir; branch başına bir olay."""
    if payload.get("eventType") not in (None, "git.push"):
        return []
    resource = payload.get("resource") or {}
    repository = resource.get("repository") or {}
    repo_id = str(repository.get("id") or "")
    if not repo_id:
        return []
    project = str((repository.get("project") or {}).get("name") or "")
    events: list[PushEvent] = []
    for update in resource.get("refUpdates") or []:
        name = str(update.get("name") or "")
        if not name.startswith("refs/heads/"):
            continue
        events.append(
            PushEvent(
                external_id=repo_id,
                repo_name=str(repository.get("name") or ""),
                project=project,
                branch=name.removeprefix("refs/heads/"),
                old_commit=str(update.get("oldObjectId") or ""),
                new_commit=str(update.get("newObjectId") or ""),
            )
        )
    return events


# Eski ad; testler ve okunabilirlik için korunur.
parse_push = parse_azure_push


def verify_secret(
    expected: str | None,
    header_secret: str | None,
    authorization: str | None,
    query_secret: str | None,
) -> bool:
    """Header, Basic auth şifresi ya da ?secret= — biri eşleşsin yeter.

    Sır yapılandırılmamışsa her istek reddedilir: açık bir webhook, herkesin
    index işi tetikleyebilmesi demek.
    """
    if not expected:
        return False
    candidates = [header_secret, query_secret]
    if authorization and authorization.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(authorization.split(" ", 1)[1]).decode("utf-8")
            candidates.append(decoded.split(":", 1)[1] if ":" in decoded else decoded)
        except (ValueError, UnicodeDecodeError):
            pass
    return any(candidate and hmac.compare_digest(candidate, expected) for candidate in candidates)


# ------------------------------------------------------------------ github


def parse_github_push(payload: dict[str, Any]) -> list[PushEvent]:
    """GitHub push olayı tek ref taşır. Branch silme (`deleted`) iş açmaz."""
    ref = str(payload.get("ref") or "")
    if not ref.startswith("refs/heads/") or payload.get("deleted") is True:
        return []
    repository = payload.get("repository") or {}
    repo_id = repository.get("id")
    if repo_id is None:
        return []
    owner = (repository.get("owner") or {}).get("login") or (repository.get("owner") or {}).get(
        "name"
    )
    return [
        PushEvent(
            external_id=str(repo_id),
            repo_name=str(repository.get("name") or ""),
            project=str(owner or ""),
            branch=ref.removeprefix("refs/heads/"),
            old_commit=str(payload.get("before") or ""),
            new_commit=str(payload.get("after") or ""),
        )
    ]


def sign_github(secret: str, body: bytes) -> str:
    """Testlerin ve dokümantasyonun kullandığı imza biçimi."""
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def verify_github_signature(expected: str | None, body: bytes, signature: str | None) -> bool:
    """GitHub, sır ayarlıysa gövdeyi HMAC-SHA256 ile imzalar; imzasız istek reddedilir."""
    if not expected or not signature or not signature.startswith("sha256="):
        return False
    return hmac.compare_digest(signature, sign_github(expected, body))
