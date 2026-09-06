"""Registering a repo: from Azure DevOps, a local directory or a plain git URL.

The repo id is the Milvus partition key and the prefix of file paths; it has to be
short, safe and stable (`azure-project-repo`). Registering the same id twice returns
the existing record — registration is idempotent.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from milvus_rag.config import Settings
from milvus_rag.db import Database
from milvus_rag.index.store import MilvusStore
from milvus_rag.log import get_logger
from milvus_rag.models import Repo
from milvus_rag.sources import git
from milvus_rag.sources.azure import AzureDevOps
from milvus_rag.sources.github import GitHub, split_full_name

log = get_logger("repos")

_SLUG_CHARS = re.compile(r"[^a-z0-9]+")
# Repo names can carry diacritics; fold the common Turkish ones into ASCII for the slug.
_TR_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")


class RepoError(ValueError):
    pass


def slugify(text: str, limit: int = 60) -> str:
    lowered = text.translate(_TR_MAP).lower()
    slug = _SLUG_CHARS.sub("-", lowered).strip("-")
    return slug[:limit].strip("-") or "repo"


class RepoService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        store: MilvusStore,
        azure: AzureDevOps | None,
        github: GitHub | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.store = store
        self.azure = azure
        self.github = github

    def register_azure(
        self, project: str, repo: str, branch: str | None = None, auto_sync: bool = True
    ) -> Repo:
        if self.azure is None:
            msg = (
                "Azure DevOps is not configured: "
                "AZURE_DEVOPS_ORG_URL and AZURE_DEVOPS_PAT are required"
            )
            raise RepoError(msg)
        info = self.azure.get_repo(project, repo)
        if info.is_disabled:
            msg = f"{info.name} is disabled on Azure"
            raise RepoError(msg)
        chosen = branch or info.default_branch
        repo_id = slugify(f"{info.project or project}-{info.name}")
        if chosen != info.default_branch:
            repo_id = slugify(f"{repo_id}-{chosen}")
        existing = self.db.get_repo(repo_id)
        if existing is not None:
            return existing
        record = Repo(
            id=repo_id,
            name=info.name,
            provider="azure",
            branch=chosen,
            local_path=str((self.settings.repos_dir / repo_id).resolve()),
            owner=info.project or project,
            external_id=info.id,
            external_name=info.name,
            remote_url=info.remote_url,
            web_url=info.web_url,
            auto_sync=auto_sync,
        )
        log.info("azure repo kaydedildi", repo=repo_id, project=record.owner, branch=chosen)
        return self.db.upsert_repo(record)

    def register_github(
        self, full_name: str, branch: str | None = None, auto_sync: bool = True
    ) -> Repo:
        """Registers `owner/repo`. Public repos work without a token."""
        if self.github is None:
            msg = "the GitHub client is not configured"
            raise RepoError(msg)
        owner, name = split_full_name(full_name)
        info = self.github.get_repo(owner, name)
        if info.archived:
            msg = f"{info.full_name} is archived"
            raise RepoError(msg)
        if info.private and not self.github.token:
            msg = f"{info.full_name} private; GITHUB_TOKEN gerekli"
            raise RepoError(msg)
        chosen = branch or info.default_branch
        repo_id = slugify(f"{info.owner}-{info.name}")
        if chosen != info.default_branch:
            repo_id = slugify(f"{repo_id}-{chosen}")
        existing = self.db.get_repo(repo_id)
        if existing is not None:
            return existing
        record = Repo(
            id=repo_id,
            name=info.full_name,
            provider="github",
            branch=chosen,
            local_path=str((self.settings.repos_dir / repo_id).resolve()),
            owner=info.owner,
            external_id=info.id,
            external_name=info.name,
            remote_url=info.clone_url,
            web_url=info.html_url,
            auto_sync=auto_sync,
        )
        log.info("github repo kaydedildi", repo=repo_id, full_name=info.full_name, branch=chosen)
        return self.db.upsert_repo(record)

    def register_local(self, path: str, name: str | None = None, auto_sync: bool = False) -> Repo:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            msg = f"no such directory: {root}"
            raise RepoError(msg)
        label = name or root.name
        repo_id = slugify(label)
        existing = self.db.get_repo(repo_id)
        if existing is not None:
            return existing
        branch = ""
        if git.is_work_tree(root):
            try:
                branch = git.current_branch(root)
            except git.GitError:
                branch = ""
        record = Repo(
            id=repo_id,
            name=label,
            provider="local",
            branch=branch,
            local_path=str(root),
            auto_sync=auto_sync,
        )
        log.info("yerel repo kaydedildi", repo=repo_id, path=str(root))
        return self.db.upsert_repo(record)

    def register_git(
        self, url: str, branch: str = "main", name: str | None = None, auto_sync: bool = False
    ) -> Repo:
        label = name or url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        repo_id = slugify(label)
        existing = self.db.get_repo(repo_id)
        if existing is not None:
            return existing
        record = Repo(
            id=repo_id,
            name=label,
            provider="git",
            branch=branch,
            local_path=str((self.settings.repos_dir / repo_id).resolve()),
            remote_url=url,
            auto_sync=auto_sync,
        )
        log.info("git repo kaydedildi", repo=repo_id, url=url)
        return self.db.upsert_repo(record)

    def remove(self, repo_id: str) -> None:
        repo = self.db.get_repo(repo_id)
        if repo is None:
            msg = f"no such repo: {repo_id}"
            raise RepoError(msg)
        self.store.delete_repo(repo_id)
        self.db.delete_repo(repo_id)
        # Only delete what we cloned ourselves; never touch the user's own directory.
        clone = Path(repo.local_path)
        repos_dir = self.settings.repos_dir.resolve()
        if repo.provider != "local" and clone.exists() and repos_dir in clone.resolve().parents:
            shutil.rmtree(clone, ignore_errors=True)
        log.info("repo silindi", repo=repo_id)
