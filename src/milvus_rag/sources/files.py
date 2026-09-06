"""Bir çalışma kopyasını indexlenebilir dosyalara çevirir.

Git deposuysa dosya listesi `git ls-files` ile alınır: .gitignore'a saygı
duyulur ve çalışma ağacındaki (henüz commit edilmemiş) dosyalar da görünür.
Değilse dizin gezilir. Her iki yolda da aynı uzantı/boyut/ikili filtreleri
uygulanır — node_modules'ü indexlemek hem 10x yavaş hem sonuçlar kütüphane
koduyla dolar.

Dil kapsamı elle tablo değil, tree-sitter-language-pack'in grammar listesidir:
uzantı adı pack'te bir grammar adıysa (`.lua`, `.vue`, `.zig`, `.php` …) o grammar
kullanılır; adı farklı olanlar (`.ts` → typescript, `.cs` → csharp) küçük bir
takma ad tablosundan geçer ve tablo içe aktarmada pack'e karşı doğrulanır —
paket sürümü değişince çökmek yerine o uzantı düz pencereye iner. Grammar'ı
olmayan ya da bilerek verilmeyen her şey yine indexlenir (paragraf/satır
pencereleri). Üretim sistemlerinin yaptığı da bu; AST'nin ölçülen katkısı
sınırlı (README.tr.md → Ölçüm defteri, "Chunk ablasyonu"), o yüzden dil başına kural
yazılmaz.
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

# Pack'in tanıdığı grammar adları (1.15.8'de 371). Literal tipten okunur; ağ yok.
PACK_GRAMMARS: frozenset[str] = frozenset(typing.get_args(SupportedLanguage))

# Uzantı → grammar, yalnızca adları FARKLI olanlar. Aynı adlılar (".go" → go,
# ".vue" → vue) tabloya girmez, `language_for`'daki kuralla eşleşir.
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
    ".m": "objc",  # MATLAB da .m kullanır; şirket kodunda Obj-C daha olası
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
    # ".cob"/".cbl" burada değil: cobol grammar'ı dengesiz parantezli girdide takılıyor
    # (duman testi: > 300 s) → CODE_WITHOUT_GRAMMAR.
    ".j2": "jinja2",
    ".jinja": "jinja2",
    ".jinja2": "jinja2",
    ".erb": "embeddedtemplate",
    ".ejs": "embeddedtemplate",
}
GRAMMAR_ALIASES: dict[str, str] = {
    ext: grammar for ext, grammar in _ALIASES.items() if grammar in PACK_GRAMMARS
}

# Uzantısız ya da özel adlı dosyalar → grammar.
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

# Grammar'ı olsa da parser'a VERİLMEYEN kod uzantıları → lang etiketi. tree-sitter-sql
# üretilmiş dev bir grammar ve drizzle migration dosyalarında segfault veriyor
# (ölçüldü: drizzle/0000_*.sql); cobol dengesiz parantezde 300 s'yi aşıyor (duman
# testi, 2026-09-01). py-tree-sitter 0.26'da parse'a süre sınırı konamıyor
# (progress_callback yolu segfault) → tek koruma bu liste + tests/test_grammars_live.py.
# Bunlar satır/paragraf pencereleriyle chunk'lanır, yine de indexlenir: SQL Server
# prosedürleri bir .NET şirketinde iş mantığının kendisi.
CODE_WITHOUT_GRAMMAR: dict[str, str] = {".sql": "sql", ".cob": "cobol", ".cbl": "cobol"}

DOC_EXTENSIONS = frozenset({".md", ".mdx", ".markdown", ".rst", ".txt", ".adoc"})

# Bilerek düz okunan metinler: config, şema, stil, şablon. Pack'te grammar'ı olsa
# bile (json, yaml, css, html) parser'a gitmezler: ölçüldü, düz pencere retrieval'da
# eşit; parser yalnız çökme yüzeyi ekler. `language_for` bunlara None döner.
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

# Grammar'ı olsa da indexlenmeyen veri/çıktı/sır uzantıları (pack'te csv, diff, po
# grammar'ı var; içerik kod değil, hacim büyük ya da sır taşır).
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

# Uzantısı olmayan ama bilinen dosyalar.
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

# Docker imajına gömülen ve duman testinden geçen grammar'lar (tests/test_grammars_live.py).
# Pack ≥ 1.15 grammar'ı ilk kullanımda indirir; kapalı ağda ilk .kt dosyasında indirme
# denemesi istemiyoruz. Liste dışı bir grammar ağ varsa yine iner, yoksa düz pencere.
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
        "packages",  # NuGet paket klasörü
        "__snapshots__",
        # WCF/SOAP araç üretimi proxy'ler (Reference.cs): binlerce satır, cevap yok.
        "Connected Services",
        "Service References",
    }
)

# Üretilmiş / kilit / minified: hacim çok, cevap yok.
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

# JSON dosyaları çoğu zaman veri dökümü; kodun onda birinden büyüğünü alma.
MAX_JSON_BYTES = 64_000


def language_for(path: str) -> str | None:
    """Dosyanın tree-sitter grammar adı; None ise düz pencereyle chunk'lanır."""
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
