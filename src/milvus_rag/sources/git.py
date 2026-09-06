"""Thin wrappers around the git command.

The credential is never embedded in the URL and never written to the remote config: it
is passed to every command as `-c http.extraheader=AUTHORIZATION: Basic ...`. That way
neither `.git/config` nor an error message ever contains the PAT.
"""

import re
import subprocess
from pathlib import Path

_TIMEOUT_CLONE = 60 * 30
_TIMEOUT_FETCH = 60 * 10
_TIMEOUT_QUICK = 60

_SECRET = re.compile(r"(Basic|Bearer)\s+[A-Za-z0-9+/=_\-]+", re.IGNORECASE)


class GitError(RuntimeError):
    pass


def _redact(text: str) -> str:
    return _SECRET.sub(r"\1 [REDACTED]", text)


def _run(args: list[str], cwd: Path | None = None, timeout: int = _TIMEOUT_QUICK) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={"GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", "PATH": _path()},
        )
    except FileNotFoundError as error:
        msg = "git not found; it has to be on PATH"
        raise GitError(msg) from error
    except subprocess.TimeoutExpired as error:
        msg = f"git {args[0]} did not finish within {timeout} s"
        raise GitError(msg) from error
    if completed.returncode != 0:
        detail = _redact(completed.stderr.strip() or completed.stdout.strip())
        msg = f"git {args[0]} failed (code {completed.returncode}): {detail}"
        raise GitError(msg)
    return completed.stdout


def _path() -> str:
    import os

    return os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin")


def _auth_args(auth_header: str | None) -> list[str]:
    return ["-c", f"http.extraheader={auth_header}"] if auth_header else []


def is_work_tree(path: Path) -> bool:
    try:
        return _run(["rev-parse", "--is-inside-work-tree"], cwd=path).strip() == "true"
    except GitError:
        return False


def head_commit(path: Path) -> str:
    return _run(["rev-parse", "HEAD"], cwd=path).strip()


def current_branch(path: Path) -> str:
    return _run(["rev-parse", "--abbrev-ref", "HEAD"], cwd=path).strip()


def ls_files(path: Path) -> list[str]:
    """Tracked + untracked-but-not-ignored files; the state of the working tree."""
    output = _run(["ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=path)
    return [entry for entry in output.split("\0") if entry]


def clone(url: str, dest: Path, branch: str, auth_header: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            *_auth_args(auth_header),
            "clone",
            "--quiet",
            "--single-branch",
            "--branch",
            branch,
            url,
            str(dest),
        ],
        timeout=_TIMEOUT_CLONE,
    )


def fetch_and_reset(path: Path, branch: str, auth_header: str | None = None) -> str:
    """Fetches the remote branch, resets the working tree onto it; returns the new HEAD.

    `reset --hard`: the clone is a read-only derivative, there should be no local
    changes. If there are, the source of truth is the remote.
    """
    _run(
        [*_auth_args(auth_header), "fetch", "--quiet", "--prune", "origin", branch],
        cwd=path,
        timeout=_TIMEOUT_FETCH,
    )
    _run(["checkout", "--quiet", "-B", branch, f"origin/{branch}"], cwd=path)
    _run(["reset", "--quiet", "--hard", f"origin/{branch}"], cwd=path)
    return head_commit(path)
