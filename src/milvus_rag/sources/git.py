"""git komut sarmalayıcıları.

Kimlik bilgisi URL'ye gömülmez, remote config'e yazılmaz: her komuta
`-c http.extraheader=AUTHORIZATION: Basic ...` olarak geçer. Böylece `.git/config`
ve hata mesajları PAT içermez.
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
        msg = "git bulunamadı; PATH'te olmalı"
        raise GitError(msg) from error
    except subprocess.TimeoutExpired as error:
        msg = f"git {args[0]} {timeout} sn içinde bitmedi"
        raise GitError(msg) from error
    if completed.returncode != 0:
        detail = _redact(completed.stderr.strip() or completed.stdout.strip())
        msg = f"git {args[0]} başarısız (kod {completed.returncode}): {detail}"
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
    """İzlenen + izlenmeyen-ama-ignore-edilmemiş dosyalar; çalışma ağacının hali."""
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
    """Uzaktaki branch'i çeker, çalışma ağacını ona eşitler; yeni HEAD'i döner.

    `reset --hard`: klon salt-okunur bir türev, yerel değişiklik olmamalı.
    Olursa da kaynak Azure'dakidir.
    """
    _run(
        [*_auth_args(auth_header), "fetch", "--quiet", "--prune", "origin", branch],
        cwd=path,
        timeout=_TIMEOUT_FETCH,
    )
    _run(["checkout", "--quiet", "-B", branch, f"origin/{branch}"], cwd=path)
    _run(["reset", "--quiet", "--hard", f"origin/{branch}"], cwd=path)
    return head_commit(path)
