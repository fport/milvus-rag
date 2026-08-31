from milvus_rag.models import Repo
from milvus_rag.webhooks import (
    parse_github_push,
    parse_push,
    sign_github,
    verify_github_signature,
    verify_secret,
)

PUSH = {
    "eventType": "git.push",
    "resource": {
        "refUpdates": [
            {"name": "refs/heads/main", "oldObjectId": "aaa", "newObjectId": "bbb"},
            {"name": "refs/tags/v1", "oldObjectId": "0", "newObjectId": "ccc"},
        ],
        "repository": {
            "id": "guid-1",
            "name": "Backend",
            "project": {"id": "p", "name": "Platform"},
            "defaultBranch": "refs/heads/main",
            "remoteUrl": "https://dev.azure.com/org/Platform/_git/Backend",
        },
        "pushedBy": {"displayName": "dev"},
    },
}


def test_parse_push_only_branches():
    events = parse_push(PUSH)
    assert len(events) == 1
    event = events[0]
    assert event.branch == "main" and event.new_commit == "bbb" and event.external_id == "guid-1"
    assert event.project == "Platform"
    assert parse_push({"eventType": "workitem.updated"}) == []
    assert parse_push({}) == []


def test_verify_secret_paths():
    import base64

    basic = "Basic " + base64.b64encode(b"hook:s3cret").decode()
    assert verify_secret("s3cret", "s3cret", None, None)
    assert verify_secret("s3cret", None, basic, None)
    assert verify_secret("s3cret", None, None, "s3cret")
    assert not verify_secret("s3cret", "wrong", None, None)
    assert not verify_secret(None, "s3cret", None, None)  # sır yoksa hep kapalı


def test_repo_manifest_and_jobs(db):
    repo = db.upsert_repo(
        Repo(id="demo", name="demo", provider="local", branch="main", local_path="/tmp/demo")
    )
    assert repo.status == "pending" and repo.created_at

    db.set_file("demo", "a.ts", "sha-a", 3)
    db.set_file("demo", "b.ts", "sha-b", 1)
    assert db.manifest("demo") == {"a.ts": "sha-a", "b.ts": "sha-b"}
    db.delete_files("demo", ["a.ts"])
    assert db.manifest("demo") == {"b.ts": "sha-b"}

    job = db.create_job("demo", "manual", force=True)
    assert db.active_job("demo").id == job.id
    db.update_job(job.id, status="running", stats={"files_seen": 2})
    assert db.get_job(job.id).stats == {"files_seen": 2}
    assert db.fail_running_jobs("restart") == 1
    assert db.get_job(job.id).status == "failed"
    assert db.active_job("demo") is None

    assert db.record_webhook("demo", "bbb") is True
    assert db.record_webhook("demo", "bbb") is False

    db.update_repo("demo", status="ready", chunk_count=4)
    assert db.get_repo("demo").chunk_count == 4
    assert db.find_repo_by_external("nope") is None
    db.delete_repo("demo")
    assert db.get_repo("demo") is None and db.manifest("demo") == {}


def test_enrichment_cache(db):
    db.set_enrichments([("h1", "Oturumu okur."), ("h2", "Cache'e yazar.")], model="m")
    assert db.get_enrichments(["h1", "h2", "h3"], "m") == {
        "h1": "Oturumu okur.",
        "h2": "Cache'e yazar.",
    }
    assert db.get_enrichments(["h1"], "other") == {}


GITHUB_PUSH = {
    "ref": "refs/heads/main",
    "before": "aaa",
    "after": "bbb",
    "deleted": False,
    "repository": {
        "id": 123456,
        "name": "backend",
        "full_name": "acme/backend",
        "default_branch": "main",
        "owner": {"login": "acme"},
    },
}


def test_parse_github_push():
    events = parse_github_push(GITHUB_PUSH)
    assert len(events) == 1
    event = events[0]
    assert event.external_id == "123456" and event.branch == "main"
    assert event.project == "acme" and event.new_commit == "bbb"
    assert parse_github_push({**GITHUB_PUSH, "ref": "refs/tags/v1"}) == []
    assert parse_github_push({**GITHUB_PUSH, "deleted": True}) == []
    assert parse_github_push({}) == []


def test_github_signature():
    body = b'{"x": 1}'
    good = sign_github("s3cret", body)
    assert verify_github_signature("s3cret", body, good)
    assert not verify_github_signature("s3cret", body + b" ", good)
    assert not verify_github_signature("s3cret", body, "sha256=deadbeef")
    assert not verify_github_signature("s3cret", body, None)
    assert not verify_github_signature(None, body, good)


def test_find_repo_by_external_scopes_provider(db):
    db.upsert_repo(
        Repo(
            id="gh",
            name="acme/backend",
            provider="github",
            branch="main",
            local_path="/tmp/gh",
            external_id="123456",
            owner="acme",
        )
    )
    assert db.find_repo_by_external("123456", provider="github").id == "gh"
    assert db.find_repo_by_external("123456", provider="azure") is None
    assert db.find_repo_by_external("123456", branch="dev") is None
