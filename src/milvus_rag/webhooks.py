"""Push webhooks: Azure DevOps "Code pushed" and GitHub "push".

Both providers are reduced to the same `PushEvent`; the API layer matches the event
to a repo record per provider. Only a push to the tracked branch opens a job; if the
same (repo, commit) arrives twice (a retry), no job is opened.

Verification differs per provider:
- Azure: a shared secret — the Basic auth password, the `X-RAG-Webhook-Secret`
  header, or `?secret=`.
- GitHub: an HMAC-SHA256 signature of the body (`X-Hub-Signature-256`), with the same
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
    external_id: str  # the Azure repo GUID or the GitHub numeric id
    repo_name: str
    project: str  # the Azure project or the GitHub owner
    branch: str
    old_commit: str
    new_commit: str


# ------------------------------------------------------------------- azure


def parse_azure_push(payload: dict[str, Any]) -> list[PushEvent]:
    """One push can update several refs; one event per branch."""
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


# Old name; kept for the tests and for readability.
parse_push = parse_azure_push


def verify_secret(
    expected: str | None,
    header_secret: str | None,
    authorization: str | None,
    query_secret: str | None,
) -> bool:
    """A header, the Basic auth password or ?secret= — one match is enough.

    If no secret is configured every request is rejected: an open webhook means
    anyone can trigger an index job.
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
    """A GitHub push event carries one ref. Deleting a branch (`deleted`) opens no job."""
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
    """The signature format the tests and the documentation use."""
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def verify_github_signature(expected: str | None, body: bytes, signature: str | None) -> bool:
    """GitHub signs the body with HMAC-SHA256 when a secret is set; an unsigned
    request is rejected."""
    if not expected or not signature or not signature.startswith("sha256="):
        return False
    return hmac.compare_digest(signature, sign_github(expected, body))
