"""The GitHub REST client (api version 2022-11-28).

The same contract as the Azure client: repo info, the repo list, the branch head. File
content again comes from `git clone`. The token is optional — public repos clone and
read without one (the API limit is 60 requests an hour; more than enough for
get_repo/branch_head). A private repo needs `GITHUB_TOKEN` (repo → contents read).

The git credential takes the same route as actions/checkout: `Basic base64("x-access-token:TOKEN")`
is passed as an extraheader, never written to the remote config.
"""

import base64
from dataclasses import dataclass
from typing import Any

import httpx

API_VERSION = "2022-11-28"
PAGE_SIZE = 100


class GitHubError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class GitHubRepo:
    id: str  # the numeric id, as a string — matches the webhook payload
    name: str
    full_name: str
    owner: str
    default_branch: str
    clone_url: str
    html_url: str
    private: bool = False
    archived: bool = False
    size_kb: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "full_name": self.full_name,
            "owner": self.owner,
            "default_branch": self.default_branch,
            "clone_url": self.clone_url,
            "html_url": self.html_url,
            "private": self.private,
            "archived": self.archived,
            "size_kb": self.size_kb,
        }


def split_full_name(full_name: str) -> tuple[str, str]:
    """`owner/repo` → (owner, repo). Works when a URL has been pasted in too."""
    cleaned = full_name.strip().removeprefix("https://github.com/").removesuffix(".git").strip("/")
    if cleaned.count("/") != 1:
        msg = f"a GitHub repo has to be in 'owner/repo' form: {full_name!r}"
        raise GitHubError(msg)
    owner, repo = cleaned.split("/")
    return owner, repo


def git_auth_header(token: str) -> str:
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return f"AUTHORIZATION: Basic {encoded}"


class GitHub:
    def __init__(
        self,
        token: str | None = None,
        api_url: str = "https://api.github.com",
        timeout: float = 30.0,
    ) -> None:
        self.token = token or None
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.Client(base_url=api_url.rstrip("/"), headers=headers, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    # ---------------------------------------------------------------- helpers
    def _get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError as error:
            msg = f"could not reach GitHub: {error}"
            raise GitHubError(msg) from error
        if response.status_code == 401:
            msg = "GitHub 401: GITHUB_TOKEN is invalid or expired"
            raise GitHubError(msg, 401)
        if response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0":
            msg = (
                "the GitHub API rate limit is exhausted (60 requests an hour without a token); "
                "add GITHUB_TOKEN to raise it to 5000"
            )
            raise GitHubError(msg, 403)
        if response.status_code == 404:
            msg = f"GitHub 404: {path} (a private repo needs GITHUB_TOKEN)"
            raise GitHubError(msg, 404)
        if response.status_code >= 400:
            msg = f"GitHub {response.status_code}: {response.text[:300]}"
            raise GitHubError(msg, response.status_code)
        return response

    # ------------------------------------------------------------------- api
    def get_repo(self, owner: str, repo: str) -> GitHubRepo:
        return _to_repo(self._get(f"/repos/{owner}/{repo}").json())

    def branch_head(self, owner: str, repo: str, branch: str) -> str | None:
        try:
            payload = self._get(f"/repos/{owner}/{repo}/branches/{branch}").json()
        except GitHubError as error:
            if error.status == 404:
                return None
            raise
        sha = (payload.get("commit") or {}).get("sha")
        return str(sha) if sha else None

    def list_repos(self, owner: str) -> list[GitHubRepo]:
        """The repos of an org or a user. An org is tried first, a user on 404."""
        try:
            rows = self._list(f"/orgs/{owner}/repos")
        except GitHubError as error:
            if error.status != 404:
                raise
            rows = self._list(f"/users/{owner}/repos")
        return sorted((_to_repo(row) for row in rows), key=lambda repo: repo.name.lower())

    def _list(self, path: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            batch = self._get(
                path, {"per_page": PAGE_SIZE, "page": page, "sort": "full_name"}
            ).json()
            if not isinstance(batch, list):
                return items
            items.extend(batch)
            if len(batch) < PAGE_SIZE:
                return items
            page += 1

    def whoami(self) -> str | None:
        """Who the token belongs to; None when there is no token. Used for verification."""
        if not self.token:
            return None
        return str(self._get("/user").json().get("login") or "")

    def git_auth_header(self) -> str | None:
        """None when there is no token: a public repo clones anonymously."""
        return git_auth_header(self.token) if self.token else None


def _to_repo(row: dict[str, Any]) -> GitHubRepo:
    return GitHubRepo(
        id=str(row["id"]),
        name=str(row["name"]),
        full_name=str(row.get("full_name") or ""),
        owner=str((row.get("owner") or {}).get("login") or ""),
        default_branch=str(row.get("default_branch") or "main"),
        clone_url=str(row.get("clone_url") or ""),
        html_url=str(row.get("html_url") or ""),
        private=bool(row.get("private", False)),
        archived=bool(row.get("archived", False)),
        size_kb=int(row.get("size") or 0),
    )
