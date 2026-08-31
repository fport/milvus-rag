"""Azure DevOps Git REST istemcisi (api-version 7.1).

Yalnızca okuma: proje ve repo listesi, repo bilgisi, branch head'i. Dosya
içeriği REST ile değil `git clone` ile gelir — artımlı sync için çalışma ağacı
gerekir, REST'ten dosya dosya çekmek hem yavaş hem kırılgan.

Kimlik: PAT, Basic auth'ta boş kullanıcı adıyla (Azure'ın beklediği biçim).
"""

import base64
from dataclasses import dataclass
from typing import Any

import httpx

API_VERSION = "7.1"


class AzureError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class AzureProject:
    id: str
    name: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "description": self.description}


@dataclass(frozen=True, slots=True)
class AzureRepo:
    id: str
    name: str
    project: str
    default_branch: str
    remote_url: str
    web_url: str
    size: int = 0
    is_disabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "project": self.project,
            "default_branch": self.default_branch,
            "remote_url": self.remote_url,
            "web_url": self.web_url,
            "size": self.size,
            "is_disabled": self.is_disabled,
        }


def normalise_org_url(url: str) -> str:
    """`https://dev.azure.com/org/` → `https://dev.azure.com/org`."""
    cleaned = url.strip().rstrip("/")
    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"https://dev.azure.com/{cleaned}"
    return cleaned


def strip_ref(ref: str) -> str:
    """`refs/heads/main` → `main`."""
    return ref.removeprefix("refs/heads/")


def git_auth_header(pat: str) -> str:
    """git'e `-c http.extraheader=` olarak geçen değer."""
    token = base64.b64encode(f":{pat}".encode()).decode()
    return f"AUTHORIZATION: Basic {token}"


class AzureDevOps:
    def __init__(self, org_url: str, pat: str, timeout: float = 30.0) -> None:
        self.org_url = normalise_org_url(org_url)
        self.pat = pat
        self._client = httpx.Client(
            base_url=self.org_url,
            auth=httpx.BasicAuth("", pat),
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    # ---------------------------------------------------------------- helpers
    def _get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        query = {"api-version": API_VERSION, **(params or {})}
        try:
            response = self._client.get(path, params=query)
        except httpx.HTTPError as error:
            msg = f"Azure DevOps'a ulaşılamadı: {error}"
            raise AzureError(msg) from error
        if response.status_code == 401:
            msg = "Azure DevOps 401: PAT geçersiz ya da süresi dolmuş (kapsam: Code → Read)"
            raise AzureError(msg, 401)
        if response.status_code == 203:
            # Azure, auth başarısızsa 203 + HTML oturum açma sayfası döner.
            msg = (
                "Azure DevOps kimlik doğrulamayı reddetti (203); "
                "PAT ve organizasyon URL'sini kontrol et"
            )
            raise AzureError(msg, 203)
        if response.status_code == 404:
            msg = f"Azure DevOps 404: {path}"
            raise AzureError(msg, 404)
        if response.status_code >= 400:
            msg = f"Azure DevOps {response.status_code}: {response.text[:300]}"
            raise AzureError(msg, response.status_code)
        return response

    def _paged(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            query = dict(params or {})
            if token:
                query["continuationToken"] = token
            response = self._get(path, query)
            payload = response.json()
            items.extend(payload.get("value", []))
            token = response.headers.get("x-ms-continuationtoken")
            if not token:
                return items

    # ------------------------------------------------------------------- api
    def list_projects(self) -> list[AzureProject]:
        rows = self._paged("/_apis/projects", {"$top": 500})
        return sorted(
            (
                AzureProject(
                    id=str(row["id"]),
                    name=str(row["name"]),
                    description=str(row.get("description") or ""),
                )
                for row in rows
            ),
            key=lambda project: project.name.lower(),
        )

    def list_repos(self, project: str) -> list[AzureRepo]:
        rows = self._paged(f"/{project}/_apis/git/repositories")
        return sorted((_to_repo(row) for row in rows), key=lambda repo: repo.name.lower())

    def get_repo(self, project: str, repo: str) -> AzureRepo:
        """`repo` ad ya da GUID olabilir; Azure ikisini de kabul eder."""
        response = self._get(f"/{project}/_apis/git/repositories/{repo}")
        return _to_repo(response.json())

    def branch_head(self, project: str, repo_id: str, branch: str) -> str | None:
        rows = self._paged(
            f"/{project}/_apis/git/repositories/{repo_id}/refs",
            {"filter": f"heads/{branch}"},
        )
        for row in rows:
            if row.get("name") == f"refs/heads/{branch}":
                return str(row["objectId"])
        return None

    def git_auth_header(self) -> str:
        return git_auth_header(self.pat)


def _to_repo(row: dict[str, Any]) -> AzureRepo:
    return AzureRepo(
        id=str(row["id"]),
        name=str(row["name"]),
        project=str(row.get("project", {}).get("name", "")),
        default_branch=strip_ref(str(row.get("defaultBranch") or "refs/heads/main")),
        remote_url=str(row.get("remoteUrl") or ""),
        web_url=str(row.get("webUrl") or ""),
        size=int(row.get("size") or 0),
        is_disabled=bool(row.get("isDisabled", False)),
    )
