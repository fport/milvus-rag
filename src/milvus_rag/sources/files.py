"""Bir çalışma kopyasını indexlenebilir dosyalara çevirir.

Git deposuysa dosya listesi `git ls-files` ile alınır: .gitignore'a saygı
duyulur ve çalışma ağacındaki (henüz commit edilmemiş) dosyalar da görünür.
Değilse dizin gezilir. Her iki yolda da aynı uzantı/boyut/ikili filtreleri
uygulanır — node_modules'ü indexlemek hem 10x yavaş hem sonuçlar kütüphane
koduyla dolar.
"""

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from milvus_rag.index.scrub import scrub
from milvus_rag.models import Category, SourceFile
from milvus_rag.sources import git

# Uzantı → tree-sitter-language-pack grammar adı. Grammar'ı olmayan uzantı
# satır bazlı chunker'a düşer, indexlenmeye devam eder.
LANGUAGE_BY_EXT: dict[str, str] = {
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mts": "typescript",
    ".cts": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".py": "python",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".dart": "dart",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
}

# Kod sayılan ama tree-sitter'a VERİLMEYEN uzantılar. tree-sitter-sql üretilmiş
# dev bir grammar ve drizzle migration dosyalarında segfault veriyor (ölçüldü:
# drizzle/0000_*.sql). Bunlar paragraf/satır pencereleriyle chunk'lanır.
CODE_WITHOUT_GRAMMAR: dict[str, str] = {
    ".sql": "sql",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".proto": "proto",
    ".prisma": "prisma",
}

DOC_EXTENSIONS = frozenset({".md", ".mdx", ".markdown", ".rst", ".txt", ".adoc"})

# Grammar'sız ama yine de okunmaya değer metinler: config, şema, şablon.
OTHER_EXTENSIONS = frozenset(
    {
        ".json",
        ".yml",
        ".yaml",
        ".toml",
        ".ini",
        ".cfg",
        ".env.example",
        ".graphql",
        ".gql",
        ".proto",
        ".prisma",
        ".xml",
        ".csproj",
        ".props",
        ".targets",
        ".sln",
        ".html",
        ".vue",
        ".svelte",
        ".css",
        ".scss",
        ".tf",
        ".hcl",
        ".dockerfile",
        ".conf",
    }
)

# Uzantısı olmayan ama bilinen dosyalar.
KNOWN_FILENAMES = frozenset(
    {"Dockerfile", "Makefile", "Jenkinsfile", "Procfile", ".env.example", "CODEOWNERS"}
)

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
        "packages",  # NuGet paket klasörü
        "__snapshots__",
    }
)

# Üretilmiş / kilit / minified: hacim çok, cevap yok.
IGNORED_SUFFIXES = (
    ".min.js",
    ".min.css",
    ".map",
    ".d.ts",
    ".lock",
    "-lock.json",
    "-lock.yaml",
    ".snap",
    ".pb.go",
    ".generated.cs",
    ".g.cs",
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

# JSON dosyaları çoğu zaman veri dökümü; kodun onda birinden büyüğünü alma.
MAX_JSON_BYTES = 64_000


def language_for(path: str) -> str | None:
    return LANGUAGE_BY_EXT.get(_suffix(path))


def category_for(path: str) -> Category:
    suffix = _suffix(path)
    if suffix in DOC_EXTENSIONS:
        return "doc"
    if suffix in LANGUAGE_BY_EXT or suffix in CODE_WITHOUT_GRAMMAR:
        return "code"
    return "other"


def _suffix(path: str) -> str:
    name = PurePosixPath(path).name
    if name.endswith(".env.example"):
        return ".env.example"
    return PurePosixPath(path).suffix.lower()


def is_indexable_path(path: str, extra_extensions: frozenset[str] = frozenset()) -> bool:
    """Sadece isme bakarak karar: uzantı, klasör ve üretilmiş dosya kuralları."""
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
    return bool(suffix) and (
        suffix in LANGUAGE_BY_EXT
        or suffix in DOC_EXTENSIONS
        or suffix in OTHER_EXTENSIONS
        or suffix in extra_extensions
    )


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def list_candidate_paths(root: Path) -> list[str]:
    """Repo kökünden göreli, POSIX ayraçlı, sıralı yol listesi."""
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
    """Dosyayı okur; ikili, çok büyük ya da UTF-8 olmayanı atlar."""
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
    # Diskteki içerik manifest'teki sha ile uyuşmuyor: dosya son indexten sonra
    # değişmiş, aramanın verdiği satır numaraları kaymış olabilir.
    stale: bool


def normalize_path(path: str) -> str:
    """ "./src/a.ts", "/src/a.ts" → "src/a.ts": manifest'in tuttuğu biçim."""
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
    """Yalnızca manifest'teki (indexlenmiş) bir dosyadan satır aralığı okur.

    Neden manifest: ajan bu aracı aramanın verdiği yolla çağırır. Uydurduğu ya da
    index dışı bir yolu (node_modules, .env) okuyabilmesi hem sır sızdırır hem
    "o dosya var" yanılgısı üretir. Manifest dışı = "indexli değil ya da yok";
    ajanın duyması gereken cevap budur. Çıktı, index'e giren metin gibi scrub'lanır.
    """
    clean = normalize_path(path)
    expected_sha = manifest.get(clean)
    if expected_sha is None:
        msg = f"{clean} indexli değil ya da yok"
        raise FileReadError("not_indexed", msg)
    target = (root / clean).resolve()
    if root.resolve() not in target.parents or not target.is_file():
        msg = f"{clean} indexlenmiş ama artık diskte yok"
        raise FileReadError("missing", msg)
    data = target.read_bytes()
    lines = scrub(data.decode("utf-8", errors="replace")).text.splitlines()
    stop = min(end or start + default_lines - 1, len(lines), start + max_lines - 1)
    stop = max(stop, start - 1)  # aralık dosyanın sonundan sonra başlıyorsa boş dilim
    return FileSlice(
        path=clean,
        start=start,
        end=stop,
        total_lines=len(lines),
        text="\n".join(lines[start - 1 : stop]),
        stale=sha256_of(data) != expected_sha,
    )
