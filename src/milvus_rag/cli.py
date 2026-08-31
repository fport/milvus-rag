"""Komut satırı.

uv run rag serve
uv run rag azure projects
uv run rag azure repos <proje>
uv run rag add-azure <proje> <repo> [--branch main]
uv run rag add-local ~/code/my-api --name my-api
uv run rag sync <repo-id> [--force]
uv run rag search "jwt token nerede üretiliyor" [--repo <id>]
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
azure_app = typer.Typer(no_args_is_help=True, help="Azure DevOps'a göz at.")
app.add_typer(azure_app, name="azure")
github_app = typer.Typer(no_args_is_help=True, help="GitHub'a göz at.")
app.add_typer(github_app, name="github")
console = Console()


def _services() -> Any:
    from milvus_rag.services import build_services

    return build_services()


@app.callback()
def _root() -> None:
    """Kod tabanı RAG servisi."""


@app.command()
def version() -> None:
    console.print(__version__)


@app.command()
def serve(
    port: Annotated[int | None, typer.Option(help="Varsayılan RAG_PORT")] = None,
    host: str = "0.0.0.0",
    reload: bool = False,
) -> None:
    """FastAPI servisini başlat."""
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
        raise typer.BadParameter("AZURE_DEVOPS_ORG_URL ve AZURE_DEVOPS_PAT gerekli")
    table = Table("proje", "id", "açıklama")
    for project in s.azure.list_projects():
        table.add_row(project.name, project.id, project.description[:60])
    console.print(table)


@azure_app.command("repos")
def azure_repos(project: str) -> None:
    s = _services()
    if s.azure is None:
        raise typer.BadParameter("AZURE_DEVOPS_ORG_URL ve AZURE_DEVOPS_PAT gerekli")
    registered = {r.external_id: r.id for r in s.db.list_repos() if r.provider == "azure"}
    table = Table("repo", "branch", "boyut", "kayıtlı", "id")
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
    """Bir org'un ya da kullanıcının repoları (public; token varsa private de)."""
    s = _services()
    registered = {r.external_id: r.id for r in s.db.list_repos() if r.provider == "github"}
    table = Table("repo", "branch", "private", "boyut", "kayıtlı")
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
    """Kayıtlı repolar."""
    s = _services()
    table = Table("id", "sağlayıcı", "branch", "durum", "dosya", "chunk", "son commit", "son index")
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
    """Azure DevOps reposunu kaydet ve indexle."""
    s = _services()
    record = s.repos.register_azure(project, repo, branch, auto_sync=not no_auto_sync)
    console.print(f"[green]kayıtlı:[/] {record.id} ({record.remote_url})")
    if not no_index:
        _sync(s, record.id, force=False)


@app.command("add-github")
def add_github(
    repo: str,
    branch: Annotated[str | None, typer.Option()] = None,
    no_index: bool = False,
    no_auto_sync: bool = False,
) -> None:
    """GitHub reposunu (`owner/repo`) kaydet ve indexle."""
    s = _services()
    record = s.repos.register_github(repo, branch, auto_sync=not no_auto_sync)
    console.print(f"[green]kayıtlı:[/] {record.id} ({record.web_url})")
    if not no_index:
        _sync(s, record.id, force=False)


@app.command("add-local")
def add_local(
    path: Path,
    name: Annotated[str | None, typer.Option()] = None,
    no_index: bool = False,
) -> None:
    """Yerel bir dizini (git olması şart değil) kaydet ve indexle."""
    s = _services()
    record = s.repos.register_local(str(path), name)
    console.print(f"[green]kayıtlı:[/] {record.id} ({record.local_path})")
    if not no_index:
        _sync(s, record.id, force=False)


@app.command("add-git")
def add_git(
    url: str, branch: str = "main", name: Annotated[str | None, typer.Option()] = None
) -> None:
    """Herkese açık ya da kimliksiz erişilebilen bir git URL'sini kaydet."""
    s = _services()
    record = s.repos.register_git(url, branch, name)
    console.print(f"[green]kayıtlı:[/] {record.id}")
    _sync(s, record.id, force=False)


@app.command()
def sync(repo_id: str, force: bool = False) -> None:
    """Repoyu şimdi, bu süreçte senkronize et (artımlı; --force tam yeniden index)."""
    _sync(_services(), repo_id, force)


@app.command()
def remove(repo_id: str, yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Repoyu, chunk'larını ve klonunu sil."""
    if not yes and not typer.confirm(f"{repo_id} silinsin mi?"):
        raise typer.Abort()
    _services().repos.remove(repo_id)
    console.print(f"[red]silindi:[/] {repo_id}")


def _sync(s: Any, repo_id: str, force: bool) -> None:
    with console.status(f"{repo_id} indexleniyor..."):
        job = s.jobs.run_now(repo_id, force=force)
    if job.status != "done":
        console.print(f"[red]başarısız:[/] {job.error}")
        raise typer.Exit(1)
    stats = job.stats
    console.print(
        f"[green]tamam[/] dosya={stats.get('files_seen')} eklendi={stats.get('added')} "
        f"değişti={stats.get('modified')} silindi={stats.get('deleted')} "
        f"aynı={stats.get('unchanged')} chunk={stats.get('chunks_written')} "
        f"süre={stats.get('total_seconds')}s embed={stats.get('embed_seconds')}s"
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
    """Kod parçalarını ara; kanal skorlarıyla birlikte."""
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
        f"[dim]mode={response.mode} rerank={response.reranked} aday={response.candidates} "
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
    """Atıflı cevap üret (LLM gerekir)."""
    from milvus_rag.search.answer import ask as run_ask
    from milvus_rag.search.retrieve import SearchRequest

    s = _services()
    if s.llm is None:
        raise typer.BadParameter("LLM yapılandırılmamış (RAG_LLM_PROVIDER / API anahtarı)")
    with console.status("düşünüyor..."):
        result = run_ask(
            s.retriever, s.llm, SearchRequest(query=query, repo_ids=tuple(repo or ()), k=k)
        )
    console.print(result.answer)
    console.print("\n[dim]kaynaklar:[/]")
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
    """Golden set ile Recall@k / MRR ölç; sonucu evals/results/ altına yaz."""
    from milvus_rag.eval import load_golden, run_eval, save_report

    s = _services()
    cases = load_golden(golden)
    with console.status(f"{len(cases)} soru ölçülüyor..."):
        report = run_eval(s.retriever, cases, tuple(repo or ()), k, tag, mode, rerank, candidates)
    path = save_report(report, results_dir)
    table = Table(title=f"{tag} · k={k} · n={report.n}", title_justify="left")
    table.add_column("kesit")
    table.add_column("n", justify="right")
    table.add_column("recall@k", justify="right")
    table.add_column("MRR", justify="right")
    table.add_row("hepsi", str(report.n), f"{report.recall_at_k:.3f}", f"{report.mrr:.3f}")
    for kind, values in report.by_kind.items():
        table.add_row(
            kind, str(int(values["n"])), f"{values['recall@k']:.3f}", f"{values['mrr']:.3f}"
        )
    console.print(table)
    console.print(f"[dim]p50={report.p50_ms:.0f}ms p95={report.p95_ms:.0f}ms · {report.config}[/]")
    if report.misses:
        console.print(f"\n[dim]ilk {k}'de bulunamayanlar:[/]")
        for miss in report.misses:
            console.print(f"  · {miss.question}  [dim]→ {', '.join(miss.top[:3])}[/]")
    console.print(f"\n[dim]yazıldı: {path}[/]")


@app.command()
def poll() -> None:
    """Azure repolarının head'ini bir kez kontrol et; değişeni indexle."""
    s = _services()
    triggered = s.jobs.poll_once()
    if not triggered:
        console.print("değişiklik yok")
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
