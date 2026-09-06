"""The command line.

uv run rag serve
uv run rag azure projects
uv run rag azure repos <project>
uv run rag add-azure <project> <repo> [--branch main]
uv run rag add-local ~/code/my-api --name my-api
uv run rag sync <repo-id> [--force]
uv run rag search "where is the jwt token issued" [--repo <id>]
uv run rag ask "..." [--repo <id>]
uv run rag eval evals/golden.example.jsonl --repo my-api --tag v1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from milvus_rag import __version__
from milvus_rag.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
azure_app = typer.Typer(no_args_is_help=True, help="Browse Azure DevOps.")
app.add_typer(azure_app, name="azure")
github_app = typer.Typer(no_args_is_help=True, help="Browse GitHub.")
app.add_typer(github_app, name="github")
console = Console()


def _services() -> Any:
    from milvus_rag.services import build_services

    return build_services()


@app.callback()
def _root() -> None:
    """A RAG service over a codebase."""


@app.command()
def version() -> None:
    console.print(__version__)


@app.command()
def serve(
    port: Annotated[int | None, typer.Option(help="Defaults to RAG_PORT")] = None,
    host: str = "0.0.0.0",
    reload: bool = False,
) -> None:
    """Start the FastAPI service."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "milvus_rag.api:app",
        host=host,
        port=port or settings.port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


# --------------------------------------------------------------------- azure


@azure_app.command("projects")
def azure_projects() -> None:
    s = _services()
    if s.azure is None:
        raise typer.BadParameter("AZURE_DEVOPS_ORG_URL and AZURE_DEVOPS_PAT are required")
    table = Table("project", "id", "description")
    for project in s.azure.list_projects():
        table.add_row(project.name, project.id, project.description[:60])
    console.print(table)


@azure_app.command("repos")
def azure_repos(project: str) -> None:
    s = _services()
    if s.azure is None:
        raise typer.BadParameter("AZURE_DEVOPS_ORG_URL and AZURE_DEVOPS_PAT are required")
    registered = {r.external_id: r.id for r in s.db.list_repos() if r.provider == "azure"}
    table = Table("repo", "branch", "size", "registered", "id")
    for repo in s.azure.list_repos(project):
        table.add_row(
            repo.name,
            repo.default_branch,
            _human(repo.size),
            registered.get(repo.id, "-"),
            repo.id,
        )
    console.print(table)


@github_app.command("repos")
def github_repo_list(owner: str) -> None:
    """The repos of an org or a user (public; private too when a token is set)."""
    s = _services()
    registered = {r.external_id: r.id for r in s.db.list_repos() if r.provider == "github"}
    table = Table("repo", "branch", "private", "size", "registered")
    for repo in s.github.list_repos(owner):
        table.add_row(
            repo.full_name,
            repo.default_branch,
            "evet" if repo.private else "",
            _human(repo.size_kb * 1024),
            registered.get(repo.id, "-"),
        )
    console.print(table)


# --------------------------------------------------------------------- repos


@app.command()
def repos() -> None:
    """Registered repos."""
    s = _services()
    table = Table(
        "id", "provider", "branch", "status", "files", "chunks", "last commit", "last index"
    )
    for repo in s.db.list_repos():
        table.add_row(
            repo.id,
            repo.provider,
            repo.branch,
            repo.status,
            str(repo.file_count),
            str(repo.chunk_count),
            repo.last_commit[:10],
            repo.last_indexed_at[:19],
        )
    console.print(table)


@app.command("add-azure")
def add_azure(
    project: str,
    repo: str,
    branch: Annotated[str | None, typer.Option()] = None,
    no_index: bool = False,
    no_auto_sync: bool = False,
) -> None:
    """Register and index an Azure DevOps repo."""
    s = _services()
    record = s.repos.register_azure(project, repo, branch, auto_sync=not no_auto_sync)
    console.print(f"[green]registered:[/] {record.id} ({record.remote_url})")
    if not no_index:
        _sync(s, record.id, force=False)


@app.command("add-github")
def add_github(
    repo: str,
    branch: Annotated[str | None, typer.Option()] = None,
    no_index: bool = False,
    no_auto_sync: bool = False,
) -> None:
    """Register and index a GitHub repo (`owner/repo`)."""
    s = _services()
    record = s.repos.register_github(repo, branch, auto_sync=not no_auto_sync)
    console.print(f"[green]registered:[/] {record.id} ({record.web_url})")
    if not no_index:
        _sync(s, record.id, force=False)


@app.command("add-local")
def add_local(
    path: Path,
    name: Annotated[str | None, typer.Option()] = None,
    no_index: bool = False,
) -> None:
    """Register and index a local directory (it need not be a git repo)."""
    s = _services()
    record = s.repos.register_local(str(path), name)
    console.print(f"[green]registered:[/] {record.id} ({record.local_path})")
    if not no_index:
        _sync(s, record.id, force=False)


@app.command("add-git")
def add_git(
    url: str, branch: str = "main", name: Annotated[str | None, typer.Option()] = None
) -> None:
    """Register a public git URL, or any URL reachable without credentials."""
    s = _services()
    record = s.repos.register_git(url, branch, name)
    console.print(f"[green]registered:[/] {record.id}")
    _sync(s, record.id, force=False)


@app.command()
def sync(repo_id: str, force: bool = False) -> None:
    """Sync the repo now, in this process (incremental; --force does a full re-index)."""
    _sync(_services(), repo_id, force)


@app.command()
def remove(repo_id: str, yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Delete the repo, its chunks and its clone."""
    if not yes and not typer.confirm(f"{repo_id} silinsin mi?"):
        raise typer.Abort()
    _services().repos.remove(repo_id)
    console.print(f"[red]deleted:[/] {repo_id}")


def _sync(s: Any, repo_id: str, force: bool) -> None:
    with console.status(f"indexing {repo_id}..."):
        job = s.jobs.run_now(repo_id, force=force)
    if job.status != "done":
        console.print(f"[red]failed:[/] {job.error}")
        raise typer.Exit(1)
    stats = job.stats
    console.print(
        f"[green]done[/] files={stats.get('files_seen')} added={stats.get('added')} "
        f"modified={stats.get('modified')} deleted={stats.get('deleted')} "
        f"unchanged={stats.get('unchanged')} chunks={stats.get('chunks_written')} "
        f"seconds={stats.get('total_seconds')}s embed={stats.get('embed_seconds')}s"
    )


# --------------------------------------------------------------------- arama


@app.command()
def search(
    query: str,
    repo: Annotated[list[str] | None, typer.Option("--repo", "-r")] = None,
    k: Annotated[int | None, typer.Option()] = None,
    mode: Annotated[str | None, typer.Option(help="auto|dense|bm25|hybrid")] = None,
    no_rerank: bool = False,
    path_prefix: str = "",
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Search code chunks, with their channel scores."""
    from milvus_rag.search.retrieve import SearchRequest

    s = _services()
    response = s.retriever.search(
        SearchRequest(
            query=query,
            repo_ids=tuple(repo or ()),
            k=k,
            mode=mode,  # type: ignore[arg-type]
            rerank=False if no_rerank else None,
            path_prefix=path_prefix,
        )
    )
    if as_json:
        console.print_json(json.dumps(response.to_dict(), ensure_ascii=False))
        return
    console.print(
        f"[dim]mode={response.mode} rerank={response.reranked} candidates={response.candidates} "
        f"{' '.join(f'{k}={v:.0f}ms' for k, v in response.timings_ms.items())}[/]"
    )
    for hit in response.hits:
        scores = " ".join(f"{name}={value:.3f}" for name, value in hit.scores.items())
        title = f"{hit.kind} [bold]{hit.symbol}[/]" if hit.symbol else hit.kind
        where = f"{hit.path}:{hit.start_line}-{hit.end_line}"
        console.print(f"\n[cyan]{hit.rank + 1}.[/] {where}  {title}  [dim]{scores}[/]")
        preview = hit.content.strip().splitlines()[:6]
        console.print("   " + "\n   ".join(preview))


@app.command()
def ask(
    query: str,
    repo: Annotated[list[str] | None, typer.Option("--repo", "-r")] = None,
    k: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Produce a cited answer (needs an LLM)."""
    from milvus_rag.search.answer import ask as run_ask
    from milvus_rag.search.retrieve import SearchRequest

    s = _services()
    if s.llm is None:
        raise typer.BadParameter("no LLM is configured (RAG_LLM_PROVIDER / API key)")
    with console.status("thinking..."):
        result = run_ask(
            s.retriever, s.llm, SearchRequest(query=query, repo_ids=tuple(repo or ()), k=k)
        )
    console.print(result.answer)
    console.print("\n[dim]sources:[/]")
    for source in result.sources:
        console.print(
            f"  [{source['n']}] {source['path']}:{source['start_line']}-{source['end_line']}"
        )
    console.print(f"[dim]{result.model} · {result.elapsed_ms:.0f} ms[/]")


@app.command()
def eval(
    golden: Path,
    repo: Annotated[list[str] | None, typer.Option("--repo", "-r")] = None,
    tag: str = "run",
    k: int = 8,
    mode: Annotated[str | None, typer.Option(help="auto|dense|bm25|hybrid")] = None,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    candidates: Annotated[int | None, typer.Option()] = None,
    results_dir: Path = Path("evals/results"),
) -> None:
    """Measure Recall@k / MRR against a golden set; write the result under evals/results/."""
    from milvus_rag.eval import load_golden, run_eval, save_report

    s = _services()
    cases = load_golden(golden)
    with console.status(f"measuring {len(cases)} questions..."):
        report = run_eval(s.retriever, cases, tuple(repo or ()), k, tag, mode, rerank, candidates)
    path = save_report(report, results_dir)
    table = Table(title=f"{tag} · k={k} · n={report.n}", title_justify="left")
    table.add_column("slice")
    table.add_column("n", justify="right")
    table.add_column("recall@k", justify="right")
    table.add_column("MRR", justify="right")
    table.add_row("all", str(report.n), f"{report.recall_at_k:.3f}", f"{report.mrr:.3f}")
    for kind, values in report.by_kind.items():
        if "abstain" in values:
            table.add_row(kind, str(int(values["n"])), f"abstain {values['abstain']:.3f}", "—")
            continue
        table.add_row(
            kind, str(int(values["n"])), f"{values['recall@k']:.3f}", f"{values['mrr']:.3f}"
        )
    console.print(table)
    signal = f"false_weak={report.false_weak_rate:.3f}"
    if report.abstain_rate is not None:
        signal = f"abstain={report.abstain_rate:.3f} {signal}"
    cal = report.calibration
    if cal.get("positive_found_min_top_dense") is not None:
        signal += (
            f" · calibration: positive min {cal['positive_found_min_top_dense']}"
            f" / negative max {cal.get('negative_max_top_dense')}"
        )
    console.print(
        f"[dim]p50={report.p50_ms:.0f}ms p95={report.p95_ms:.0f}ms · {signal} · {report.config}[/]"
    )
    if report.misses:
        console.print(f"\n[dim]not found in the top {k} / failed to abstain:[/]")
        for miss in report.misses:
            console.print(f"  · {miss.question}  [dim]→ {', '.join(miss.top[:3])}[/]")
    console.print(f"\n[dim]written: {path}[/]")


@app.command()
def poll() -> None:
    """Check the heads of the remote repos once; index whatever moved."""
    s = _services()
    triggered = s.jobs.poll_once()
    if not triggered:
        console.print("no changes")
        return
    for item in triggered:
        _sync(s, item["repo_id"], force=False)


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.0f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


if __name__ == "__main__":
    app()
