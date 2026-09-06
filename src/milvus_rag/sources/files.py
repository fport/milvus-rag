"""Turns a working copy into indexable files.

For a git repo the file list comes from `git ls-files`: .gitignore is respected and
files in the working tree (not committed yet) show up too. Otherwise the directory is
walked. Both paths apply the same extension/size/binary filters — indexing node_modules
is 10x slower and fills the results with library code.


Language coverage is not a hand-written table, it is tree-sitter-language-pack's grammar
list: if the extension name is a grammar name in the pack (`.lua`, `.vue`, `.zig`, `.php`
…) that grammar is used; the ones whose names differ (`.ts` → typescript, `.cs` →
csharp) go through a small alias table, and that table is validated against the pack at
import time — when the package version changes, the extension falls back to plain
windows instead of crashing. Anything without a grammar, or deliberately withheld from
one, is still indexed (paragraph/line windows). Production systems do the same; the
measured contribution of the AST is limited (README → Measurement ledger, "Chunk
ablation"), which is why no rule is written per language.
"""

import hashlib
import os
import typing
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from tree_sitter_language_pack import SupportedLanguage

from milvus_rag.index.scrub import scrub
from milvus_rag.models import Category, SourceFile
from milvus_rag.sources import git

# The grammar names the pack knows (371 in 1.15.8). Read from the literal type; no network.
PACK_GRAMMARS: frozenset[str] = frozenset(typing.get_args(SupportedLanguage))

# Extension → grammar, only for the ones whose names DIFFER. Matching names (".go" → go,
# ".vue" → vue) never enter the table; they are matched by the rule in `language_for`.
_ALIASES: dict[str, str] = {
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".rb": "ruby",
    ".rake": "ruby",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".cshtml": "razor",
    ".h": "c",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".m": "objc",  # MATLAB uses .m too; in company code Obj-C is more likely
    ".mm": "objc",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".fs": "fsharp",
    ".fsx": "fsharp",
    ".fsi": "fsharp",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".pl": "perl",
    ".pm": "perl",
    ".jl": "julia",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
    ".groovy": "groovy",
    ".gvy": "groovy",
    ".gradle": "groovy",
    ".gql": "graphql",
    ".sol": "solidity",
    ".mk": "make",
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".bzl": "starlark",
    ".bazel": "starlark",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".vert": "glsl",
    ".frag": "glsl",
    ".sv": "systemverilog",
    ".svh": "systemverilog",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".pas": "pascal",
    ".adb": "ada",
    ".ads": "ada",
    ".cls": "apex",
    ".trigger": "apex",
    # ".cob"/".cbl" are not here: the cobol grammar hangs on unbalanced-paren input
    # (smoke test: > 300 s) → CODE_WITHOUT_GRAMMAR.
    ".j2": "jinja2",
    ".jinja": "jinja2",
    ".jinja2": "jinja2",
    ".erb": "embeddedtemplate",
    ".ejs": "embeddedtemplate",
}
GRAMMAR_ALIASES: dict[str, str] = {
    ext: grammar for ext, grammar in _ALIASES.items() if grammar in PACK_GRAMMARS
}

# Files with no extension, or with a special name → grammar.
GRAMMAR_BY_FILENAME: dict[str, str] = {
    name: grammar
    for name, grammar in {
        "Dockerfile": "dockerfile",
        "Makefile": "make",
        "Jenkinsfile": "groovy",
        "go.mod": "gomod",
        "BUILD": "starlark",
        "WORKSPACE": "starlark",
    }.items()
    if grammar in PACK_GRAMMARS
}

# Code extensions that have a grammar but are NOT handed to the parser → a lang label.
# tree-sitter-sql is a huge generated grammar and segfaults on drizzle migration files
# (measured: drizzle/0000_*.sql); cobol exceeds 300 s on unbalanced parens (smoke test,
# 2026-09-01). py-tree-sitter 0.26 offers no time limit on a parse (the progress_callback
# path segfaults) → this list plus tests/test_grammars_live.py is the only guard.
# These are still indexed, chunked with line/paragraph windows: SQL Server procedures
# are the business logic itself in a .NET shop.
CODE_WITHOUT_GRAMMAR: dict[str, str] = {".sql": "sql", ".cob": "cobol", ".cbl": "cobol"}

DOC_EXTENSIONS = frozenset({".md", ".mdx", ".markdown", ".rst", ".txt", ".adoc"})

# Text we deliberately read flat: config, schema, style, template. Even when the pack has
# a grammar (json, yaml, css, html) they never reach the parser: measured, plain windows
# are equal on retrieval and the parser only adds crash surface. `language_for` returns
# None for these.
OTHER_EXTENSIONS = frozenset(
    {
        ".json",
        ".json5",
        ".jsonc",
        ".yml",
        ".yaml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".properties",
        ".env.example",
        ".xml",
        ".csproj",
        ".vbproj",
        ".fsproj",
        ".sqlproj",
        ".props",
        ".targets",
        ".sln",
        ".nuspec",
        ".config",
        ".xaml",
        ".plist",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".hbs",
        ".handlebars",
        ".mustache",
        ".pug",
        ".haml",
        ".slim",
        ".jsp",
        ".aspx",
        ".ascx",
        ".master",
    }
)

# Data/output/secret extensions that are never indexed even when a grammar exists (the
# pack has csv, diff and po grammars; the content is not code, the volume is large, or it
# carries secrets).
NEVER_INDEX_EXTENSIONS = frozenset(
    {
        ".csv",
        ".tsv",
        ".diff",
        ".patch",
        ".po",
        ".pot",
        ".log",
        ".ipynb",
        ".svg",
        ".pem",
        ".crt",
        ".key",
        ".pfx",
        ".p12",
    }
)

# Files with no extension but a known name.
KNOWN_FILENAMES = frozenset(
    {
        "Dockerfile",
        "Makefile",
        "Jenkinsfile",
        "Procfile",
        ".env.example",
        "CODEOWNERS",
        "BUILD",
        "WORKSPACE",
    }
)

# Grammars baked into the Docker image that pass the smoke test (tests/test_grammars_live.py).
# Pack ≥ 1.15 downloads a grammar on first use; on an offline host we do not want that
# attempt on the first .kt file. A grammar outside the list still downloads if there is a
# network, and falls back to plain windows if there is not.
PREFETCH_GRAMMARS: frozenset[str] = (
    frozenset({*GRAMMAR_ALIASES.values(), *GRAMMAR_BY_FILENAME.values()})
    | frozenset(
        {
            "c",
            "cpp",
            "go",
            "java",
            "php",
            "swift",
            "scala",
            "dart",
            "lua",
            "vb",
            "vue",
            "svelte",
            "astro",
            "razor",
            "twig",
            "liquid",
            "blade",
            "graphql",
            "proto",
            "prisma",
            "hcl",
            "nix",
            "zig",
            "nim",
            "elm",
            "r",
            "cmake",
        }
    )
) & PACK_GRAMMARS

IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "bower_components",
        "vendor",
        "dist",
        "build",
        "out",
        "target",
        "bin",
        "obj",
        "coverage",
        ".next",
        ".nuxt",
        ".turbo",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        ".gradle",
        ".dart_tool",
        "Pods",
        "DerivedData",
        "packages",  # the NuGet package folder
        "__snapshots__",
        # WCF/SOAP tool-generated proxies (Reference.cs): thousands of lines, no answers.
        "Connected Services",
        "Service References",
    }
)

# Generated / lock / minified: a lot of volume, no answers.
IGNORED_SUFFIXES = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".chunk.js",
    ".map",
    ".d.ts",
    ".lock",
    "-lock.json",
    "-lock.yaml",
    ".snap",
    ".pb.go",
    ".generated.cs",
    ".g.cs",
    ".g.i.cs",
    ".Designer.cs",
    ".designer.cs",
    ".g.dart",
    ".freezed.dart",
)
IGNORED_FILENAMES = frozenset(
    {
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "bun.lockb",
        "bun.lock",
        "uv.lock",
        "poetry.lock",
        "Cargo.lock",
        "composer.lock",
        "Gemfile.lock",
        "go.sum",
        "packages.lock.json",
    }
)

# JSON files are usually data dumps; do not take one bigger than a tenth of the code limit.
MAX_JSON_BYTES = 64_000


def language_for(path: str) -> str | None:
    """The file's tree-sitter grammar name; None means it is chunked with plain windows."""
    name = PurePosixPath(path).name
    if name in GRAMMAR_BY_FILENAME:
        return GRAMMAR_BY_FILENAME[name]
    suffix = _suffix(path)
    if (
        not suffix
        or suffix in DOC_EXTENSIONS
        or suffix in OTHER_EXTENSIONS
        or suffix in CODE_WITHOUT_GRAMMAR
        or suffix in NEVER_INDEX_EXTENSIONS
    ):
        return None
    if suffix in GRAMMAR_ALIASES:
        return GRAMMAR_ALIASES[suffix]
    grammar = suffix[1:]
    return grammar if grammar in PACK_GRAMMARS else None


def category_for(path: str) -> Category:
    suffix = _suffix(path)
    if suffix in DOC_EXTENSIONS:
        return "doc"
    if suffix in CODE_WITHOUT_GRAMMAR or language_for(path) is not None:
        return "code"
    return "other"


def _suffix(path: str) -> str:
    name = PurePosixPath(path).name
    if name.endswith(".env.example"):
        return ".env.example"
    return PurePosixPath(path).suffix.lower()


def is_indexable_path(path: str, extra_extensions: frozenset[str] = frozenset()) -> bool:
    """Decided from the name alone: extension, folder and generated-file rules."""
    posix = PurePosixPath(path)
    if IGNORED_DIRS & set(posix.parts[:-1]):
        return False
    name = posix.name
    if name in IGNORED_FILENAMES:
        return False
    lowered = name.lower()
    if any(lowered.endswith(suffix) for suffix in IGNORED_SUFFIXES):
        return False
    if name in KNOWN_FILENAMES:
        return True
    suffix = _suffix(path)
    if not suffix or suffix in NEVER_INDEX_EXTENSIONS:
        return False
    return (
        suffix in DOC_EXTENSIONS
        or suffix in OTHER_EXTENSIONS
        or suffix in CODE_WITHOUT_GRAMMAR
        or suffix in extra_extensions
        or language_for(path) is not None
    )


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def list_candidate_paths(root: Path) -> list[str]:
    """Paths relative to the repo root, POSIX separators, sorted."""
    root = root.resolve()
    if git.is_work_tree(root):
        paths = git.ls_files(root)
    else:
        paths = []
        for current, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            relative_dir = Path(current).relative_to(root)
            for filename in filenames:
                paths.append((relative_dir / filename).as_posix())
    return sorted(path for path in paths if not path.startswith(".git/"))


def read_source_file(root: Path, path: str, max_bytes: int) -> SourceFile | None:
    """Reads the file; skips binary, oversized and non-UTF-8 ones."""
    full = root / path
    try:
        if full.is_symlink() or not full.is_file():
            return None
        size = full.stat().st_size
    except OSError:
        return None
    limit = min(max_bytes, MAX_JSON_BYTES) if _suffix(path) == ".json" else max_bytes
    if size == 0 or size > limit:
        return None
    try:
        data = full.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text.strip():
        return None
    return SourceFile(
        path=path,
        text=text,
        sha=sha256_of(data),
        lang=language_for(path) or CODE_WITHOUT_GRAMMAR.get(_suffix(path), ""),
        category=category_for(path),
    )


def iter_source_files(
    root: Path, max_bytes: int, extra_extensions: frozenset[str] = frozenset()
) -> Iterator[SourceFile]:
    for path in list_candidate_paths(root):
        if not is_indexable_path(path, extra_extensions):
            continue
        source = read_source_file(root, path, max_bytes)
        if source is not None:
            yield source


def parse_extra_extensions(raw: str) -> frozenset[str]:
    items = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return frozenset(item if item.startswith(".") else f".{item}" for item in items)


# ------------------------------------------------------------ dosya okuma


class FileReadError(ValueError):
    """`reason`: "not_indexed" (yol manifest'te yok) ya da "missing" (indexli ama diskte yok)."""

    def __init__(self, reason: Literal["not_indexed", "missing"], message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class FileSlice:
    path: str
    start: int
    end: int
    total_lines: int
    text: str
    # The content on disk does not match the sha in the manifest: the file changed after
    # the last index, so the line numbers search returned may have shifted.
    stale: bool


def normalize_path(path: str) -> str:
    """ "./src/a.ts", "/src/a.ts" → "src/a.ts": the form the manifest keeps."""
    return str(PurePosixPath(path.strip().lstrip("/")))


def read_indexed_slice(
    root: Path,
    manifest: dict[str, str],
    path: str,
    start: int = 1,
    end: int | None = None,
    default_lines: int = 200,
    max_lines: int = 400,
) -> FileSlice:
    """Reads a line range only from a file that is in the manifest (i.e. indexed).

    Why the manifest: an agent calls this tool with a path search gave it. Letting it read
    an invented path, or one outside the index (node_modules, .env), both leaks secrets and
    creates the illusion that "the file exists". Outside the manifest = "not indexed or does
    not exist"; that is the answer the agent needs to hear. The output is scrubbed like the
    text that goes into the index.
    """
    clean = normalize_path(path)
    expected_sha = manifest.get(clean)
    if expected_sha is None:
        msg = f"{clean} is not indexed or does not exist"
        raise FileReadError("not_indexed", msg)
    target = (root / clean).resolve()
    if root.resolve() not in target.parents or not target.is_file():
        msg = f"{clean} was indexed but is no longer on disk"
        raise FileReadError("missing", msg)
    data = target.read_bytes()
    lines = scrub(data.decode("utf-8", errors="replace")).text.splitlines()
    stop = min(end or start + default_lines - 1, len(lines), start + max_lines - 1)
    stop = max(stop, start - 1)  # an empty slice when the range starts past the end of the file
    return FileSlice(
        path=clean,
        start=start,
        end=stop,
        total_lines=len(lines),
        text="\n".join(lines[start - 1 : stop]),
        stale=sha256_of(data) != expected_sha,
    )
