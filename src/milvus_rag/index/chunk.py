"""Kaynak dosyayı retrieval birimlerine böler.

Kod için chunk = kod birimi: fonksiyon, sınıf, metod, interface, type, export
edilmiş const. Sınırlar gerçek bir parser'dan (tree-sitter) gelir; süslü parantez
saymak string içindeki `}` ile bozulur. Dört kural:

1. Büyük sınıf/namespace → üyelerine bölünür; başlık + alanlar ayrı chunk olur.
2. Küçük şeyler (tek satır type, kısa const, import bloğu, JSDoc) komşusuyla
   birleşir; ~MIN byte altına inilmez.
3. Hiçbir chunk ~MAX byte'ı geçmez (≈500 token). Model 8192 token alsa da uzun
   chunk'ın embedding'i "ortalama"ya döner, hiçbir şeyi iyi temsil etmez. Tek
   başına MAX'ı aşan bir fonksiyon satır pencerelerine bölünür, sembolünü korur.
4. `text` temiz kalır (LLM'e ve atıfa giden); "ben neyim, nerede yaşıyorum"
   başlığı yalnızca embed/BM25 metnine eklenir.

Markdown başlık yoluna göre, grammar'sız metin paragraf pencerelerine bölünür.
"""

from __future__ import annotations

import re
import threading
from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from milvus_rag.models import ChunkRecord
from milvus_rag.sources.files import category_for, language_for

if TYPE_CHECKING:
    from tree_sitter import Node

# Grammar'lar arası birleşik "kod birimi" düğüm türleri. Bir düğüm burada
# yoksa dolgudur (import, yorum, top-level ifade) ve komşusuna yapışır.
UNIT_TYPES = frozenset(
    {
        # JS / TS
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "abstract_class_declaration",
        "method_definition",
        "method_signature",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "lexical_declaration",
        "variable_declaration",
        "public_field_definition",
        "internal_module",  # TS namespace
        "module",  # TS declare module / Ruby module
        # Python
        "function_definition",
        "class_definition",
        "decorated_definition",
        # Go
        "method_declaration",
        "type_declaration",
        "const_declaration",
        "var_declaration",
        # Java / C# / Kotlin
        "constructor_declaration",
        "record_declaration",
        "struct_declaration",
        "namespace_declaration",
        "file_scoped_namespace_declaration",
        "property_declaration",
        "field_declaration",
        "delegate_declaration",
        "event_field_declaration",
        "object_declaration",
        "annotation_type_declaration",
        # Rust
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "trait_item",
        "mod_item",
        "type_item",
        "const_item",
        "static_item",
        "macro_definition",
        # Ruby
        "class",
        "method",
        "singleton_method",
        # C / C++
        "struct_specifier",
        "class_specifier",
        "declaration",
        # PHP
        "trait_declaration",
    }
)

# Sığmazsa üyelerine inilen kapsayıcılar.
CONTAINER_TYPES = frozenset(
    {
        "class_declaration",
        "abstract_class_declaration",
        "class_definition",
        "class_specifier",
        "interface_declaration",
        "enum_declaration",
        "struct_declaration",
        "record_declaration",
        "namespace_declaration",
        "file_scoped_namespace_declaration",
        "internal_module",
        "module",
        "impl_item",
        "trait_item",
        "mod_item",
        "object_declaration",
        "class",
        "trait_declaration",
    }
)

# Dış sarmalayıcı → içindeki asıl bildirimin alanı.
WRAPPER_FIELDS: dict[str, str] = {
    "export_statement": "declaration",
    "decorated_definition": "definition",
    "ambient_declaration": "declaration",
}

BODY_TYPES = frozenset(
    {
        "class_body",
        "declaration_list",
        "block",
        "body_statement",
        "enum_body",
        "interface_body",
        "object_type",
        "statement_block",
        "field_declaration_list",
        "impl_body",
    }
)

IMPORT_TYPES = frozenset(
    {
        "import_statement",
        "import_from_statement",
        "import_declaration",
        "using_directive",
        "use_declaration",
        "import_header",
        "require_call",
    }
)

# Düğüm türü → normalize edilmiş tür adı (başlıkta ve `kind` alanında görünür).
KIND_BY_TYPE: dict[str, str] = {
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "function_definition": "function",
    "function_item": "function",
    "method_definition": "method",
    "method_declaration": "method",
    "method_signature": "method",
    "method": "method",
    "singleton_method": "method",
    "constructor_declaration": "constructor",
    "class_declaration": "class",
    "abstract_class_declaration": "class",
    "class_definition": "class",
    "class_specifier": "class",
    "class": "class",
    "record_declaration": "record",
    "struct_declaration": "struct",
    "struct_item": "struct",
    "struct_specifier": "struct",
    "interface_declaration": "interface",
    "trait_item": "trait",
    "trait_declaration": "trait",
    "impl_item": "impl",
    "type_alias_declaration": "type",
    "type_declaration": "type",
    "type_item": "type",
    "enum_declaration": "enum",
    "enum_item": "enum",
    "namespace_declaration": "namespace",
    "file_scoped_namespace_declaration": "namespace",
    "internal_module": "namespace",
    "module": "module",
    "mod_item": "module",
    "object_declaration": "object",
    "lexical_declaration": "const",
    "variable_declaration": "const",
    "const_declaration": "const",
    "const_item": "const",
    "static_item": "const",
    "var_declaration": "var",
    "public_field_definition": "field",
    "field_declaration": "field",
    "property_declaration": "property",
    "event_field_declaration": "event",
    "delegate_declaration": "delegate",
    "macro_definition": "macro",
    "declaration": "declaration",
}

HASH_COMMENT_LANGS = frozenset({"python", "ruby", "bash", "yaml", "toml", "r", "perl"})

MAX_IMPORTS = 10
MAX_TOKENS = 400
LINE_WINDOW_OVERLAP = 3

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL_LOWER_UPPER = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_BLANK_LINES = re.compile(r"\n\s*\n")


class _Lines:
    """Byte ofseti → 0 tabanlı satır; tree-sitter'ın start_point/end_point'i yerine.

    Point erişimi py-tree-sitter 0.26 + language-pack 1.15 ile uzun süreçte
    segfault verdi (ölçüldü: 39. dosyada, tek başına aynı dosya geçiyor). Satır
    numarasını kaynaktan kendimiz hesaplıyoruz; bisect ile O(log n).
    """

    def __init__(self, data: bytes) -> None:
        starts = [0]
        offset = data.find(b"\n")
        while offset != -1:
            starts.append(offset + 1)
            offset = data.find(b"\n", offset + 1)
        self._starts = starts

    def row(self, offset: int) -> int:
        return bisect_right(self._starts, max(offset, 0)) - 1


@dataclass(frozen=True, slots=True)
class Piece:
    """Parser'dan çıkan, henüz paketlenmemiş aralık."""

    start: int
    end: int
    start_row: int
    end_row: int
    symbol: str = ""
    kind: str = ""
    parent: str = ""
    unit: bool = False


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    max_bytes: int = 2000
    min_bytes: int = 200


# ----------------------------------------------------------------- public API


def chunk_file(path: str, text: str, config: ChunkerConfig | None = None) -> list[ChunkRecord]:
    """Dosyayı chunk'lara böler. Her dosya için en az bir chunk döner (boş değilse)."""
    config = config or ChunkerConfig()
    if not text.strip():
        return []
    category = category_for(path)
    language = language_for(path)
    if category == "doc":
        return _chunk_markdown(path, text, config)
    if language:
        records = _chunk_code(path, text, language, config)
        if records:
            return records
    return _chunk_plain(path, text, config)


def split_identifiers(text: str, limit: int = MAX_TOKENS) -> str:
    """camelCase / snake_case / PascalCase adları BM25 için alt kelimelere böler.

    `handleAuthCallback` standart analyzer'da tek token kalır; "auth callback
    nerede" sorusu ona değmez. Alt kelimeler küçük harfle, sıralı, tekil.
    """
    parts: set[str] = set()
    for word in _IDENTIFIER.findall(text):
        spaced = _CAMEL_ACRONYM.sub(r"\1 \2", _CAMEL_LOWER_UPPER.sub(r"\1 \2", word))
        for part in spaced.replace("_", " ").split():
            if len(part) > 1 and not part.isdigit():
                parts.add(part.lower())
        if len(parts) >= limit:
            break
    return " ".join(sorted(parts))


def build_indexed_text(record: ChunkRecord, context: str = "") -> str:
    """Embed ve BM25'e giden metin: başlık → (LLM açıklaması) → semboller → kod → alt kelimeler."""
    sections = [record.header]
    if context:
        sections.append(context)
    if record.symbols:
        sections.append(" ".join(record.symbols))
    sections.append(record.text)
    tokens = split_identifiers(f"{record.header}\n{record.text}")
    if tokens:
        sections.append(tokens)
    return "\n".join(section for section in sections if section)


# --------------------------------------------------------------------- code


_PARSE_LOCK = threading.Lock()
# Language nesneleri süreç boyunca burada yaşar. tree-sitter-language-pack'in
# get_parser()'ı Language'ı parser'a zayıf bağlıyor; Language erken serbest
# kalınca ağaç üstündeki child_by_field_name() sarkan işaretçiyle segfault
# veriyor (ölçüldü: aynı dosya tek başına geçiyor, 40. dosyada çöküyordu).
_LANGUAGES: dict[str, Any] = {}
_PARSERS: dict[str, Any] = {}


def _parser_for(language: str) -> Any | None:
    parser = _PARSERS.get(language)
    if parser is not None:
        return parser
    try:
        from tree_sitter import Parser
        from tree_sitter_language_pack import get_language

        grammar = get_language(language)
        parser = Parser(grammar)
    except Exception:
        return None
    _LANGUAGES[language] = grammar
    _PARSERS[language] = parser
    return parser


def _chunk_code(path: str, text: str, language: str, config: ChunkerConfig) -> list[ChunkRecord]:
    parser = _parser_for(language)
    if parser is None:
        return []

    data = text.encode("utf-8")
    with _PARSE_LOCK:
        tree = parser.parse(data)
    root = tree.root_node
    if root is None or not root.children:
        return []

    imports = _imports(root, data)
    lines = _Lines(data)
    pieces: list[Piece] = []
    for node in root.children:
        pieces.extend(_pieces_of(node, data, lines, config, parent=""))
    if not pieces:
        return []

    comment = "#" if language in HASH_COMMENT_LANGS else "//"
    header_for = _code_header_factory(path, comment, imports)
    return _pack(pieces, data, config, header_for)


def _pieces_of(
    node: Node, data: bytes, lines: _Lines, config: ChunkerConfig, parent: str
) -> list[Piece]:
    inner = _unwrap(node)
    size = node.end_byte - node.start_byte
    symbol = _name_of(inner, data)
    kind = KIND_BY_TYPE.get(inner.type, "")
    is_unit = inner.type in UNIT_TYPES

    if is_unit and inner.type in CONTAINER_TYPES and size > config.max_bytes:
        body = _body_of(inner)
        members = [child for child in body.children if child.is_named] if body else []
        if body is not None and members:
            qualified = f"{parent}.{symbol}" if parent and symbol else (symbol or parent)
            header_end = body.start_byte + 1 if body.type != "block" else body.start_byte
            pieces = [
                Piece(
                    start=node.start_byte,
                    end=max(header_end, node.start_byte),
                    start_row=lines.row(node.start_byte),
                    end_row=lines.row(header_end),
                    symbol=symbol,
                    kind=kind or "class",
                    parent=parent,
                    unit=True,
                )
            ]
            for member in members:
                pieces.extend(_pieces_of(member, data, lines, config, parent=qualified))
            return pieces

    if not is_unit:
        symbol = _assignment_name(inner, data)
        kind = "const" if symbol else ""
    return [
        Piece(
            start=node.start_byte,
            end=node.end_byte,
            start_row=lines.row(node.start_byte),
            end_row=lines.row(max(node.end_byte - 1, node.start_byte)),
            symbol=symbol,
            kind=kind,
            parent=parent,
            unit=is_unit or bool(symbol),
        )
    ]


def _unwrap(node: Node) -> Node:
    field = WRAPPER_FIELDS.get(node.type)
    if field is None:
        return node
    inner = node.child_by_field_name(field)
    return _unwrap(inner) if inner is not None else node


def _body_of(node: Node) -> Node | None:
    body = node.child_by_field_name("body")
    if body is not None:
        return body
    for child in node.children:
        if child.type in BODY_TYPES or child.type.endswith("_body"):
            return child
    # C# file-scoped namespace: üyeler doğrudan çocuk. Adı ve noktalı virgülü
    # atlayıp gövde olarak düğümün kendisini kullan.
    if node.type == "file_scoped_namespace_declaration":
        return node
    return None


def _name_of(node: Node, data: bytes) -> str:
    name = node.child_by_field_name("name")
    if name is not None:
        return _text(name, data)
    if node.type == "impl_item":
        target = node.child_by_field_name("type")
        trait = node.child_by_field_name("trait")
        if target is None:
            return ""
        return (
            f"impl {_text(trait, data)} for {_text(target, data)}"
            if trait is not None
            else f"impl {_text(target, data)}"
        )
    if node.type in (
        "lexical_declaration",
        "variable_declaration",
        "const_declaration",
        "var_declaration",
        "type_declaration",
        "field_declaration",
        "property_declaration",
        "event_field_declaration",
        "declaration",
    ):
        for child in node.children:
            if not child.is_named:
                continue
            found = _name_of(child, data)
            if found:
                return found
        return ""
    if node.type in ("variable_declarator", "const_spec", "type_spec", "init_declarator"):
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "field_identifier"):
                return _text(child, data)
    return ""


def _assignment_name(node: Node, data: bytes) -> str:
    """Python `FOO = 1`, JS `module.exports = ...` gibi top-level atamalar."""
    if node.type != "expression_statement" or not node.children:
        return ""
    first = node.children[0]
    if first.type in ("assignment", "assignment_expression"):
        left = first.child_by_field_name("left")
        if left is not None and left.type == "identifier":
            return _text(left, data)
    return ""


def _imports(root: Node, data: bytes) -> list[str]:
    found: list[str] = []
    for node in root.children:
        candidates = [node]
        if node.type == "import_declaration" and node.children:
            # Go: import ( "a" "b" ) — spec'ler alt düğümde.
            candidates = [child for child in node.children if child.is_named] or [node]
        for candidate in candidates:
            if candidate.type not in IMPORT_TYPES and candidate.type != "import_spec":
                continue
            source = (
                candidate.child_by_field_name("source")
                or candidate.child_by_field_name("path")
                or candidate.child_by_field_name("name")
                or candidate.child_by_field_name("module_name")
                or candidate.child_by_field_name("argument")
            )
            raw = _text(source, data) if source is not None else _text(candidate, data)
            cleaned = _clean_import(raw)
            if cleaned and cleaned not in found:
                found.append(cleaned)
            if len(found) >= MAX_IMPORTS:
                return found
    return found


def _clean_import(raw: str) -> str:
    text = raw.strip().strip(";").strip()
    for prefix in ("import ", "from ", "using ", "use ", "require"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = text.strip("\"'()").strip()
    return text[:60]


def _code_header_factory(
    path: str, comment: str, imports: list[str]
) -> Callable[[str, str, str, int, int], str]:
    import_line = f"{comment} imports: {', '.join(imports)}" if imports else ""

    def build(symbol: str, kind: str, parent: str, part: int, parts: int) -> str:
        lines = [f"{comment} file: {path}"]
        if symbol:
            where = f" (in {parent})" if parent else ""
            suffix = f" [part {part}/{parts}]" if parts > 1 else ""
            lines.append(f"{comment} {kind or 'symbol'}: {symbol}{where}{suffix}")
        elif parent:
            lines.append(f"{comment} in: {parent}")
        if import_line:
            lines.append(import_line)
        return "\n".join(lines)

    return build


def _pack(
    pieces: list[Piece],
    data: bytes,
    config: ChunkerConfig,
    header_for: Callable[[str, str, str, int, int], str],
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    buffer: list[Piece] = []

    def span(items: list[Piece]) -> int:
        return items[-1].end - items[0].start

    def emit(items: list[Piece], part: int = 1, parts: int = 1) -> None:
        start, end = items[0].start, items[-1].end
        body = data[start:end].decode("utf-8", errors="replace")
        if not body.strip():
            return
        lead = next((item for item in items if item.symbol), None)
        symbol = lead.symbol if lead else ""
        kind = lead.kind if lead and lead.kind else ("block" if not lead else "symbol")
        parent = lead.parent if lead else items[0].parent
        declared = tuple(dict.fromkeys(item.symbol for item in items if item.symbol))
        records.append(
            ChunkRecord(
                ordinal=len(records),
                text=body,
                start_line=items[0].start_row + 1,
                end_line=items[-1].end_row + 1,
                symbol=symbol,
                parent_symbol=parent,
                kind=kind,
                header=header_for(symbol, kind, parent, part, parts),
                symbols=declared,
            )
        )

    def flush() -> None:
        if buffer:
            emit(list(buffer))
            buffer.clear()

    for piece in pieces:
        size = piece.end - piece.start
        if size > config.max_bytes:
            flush()
            windows = _line_windows(piece, data, config.max_bytes)
            for index, window in enumerate(windows, start=1):
                emit([window], part=index, parts=len(windows))
            continue
        if buffer and span([*buffer, piece]) > config.max_bytes:
            flush()
        buffer.append(piece)
        if piece.unit and span(buffer) >= config.min_bytes:
            flush()
    flush()
    return records


def _line_windows(piece: Piece, data: bytes, max_bytes: int) -> list[Piece]:
    """MAX'ı aşan tek birimi satır pencerelerine böler; sembol ve tür korunur."""
    segment = data[piece.start : piece.end]
    lines = segment.split(b"\n")
    windows: list[Piece] = []
    offset = piece.start
    row = piece.start_row
    index = 0
    while index < len(lines):
        start_offset, start_row = offset, row
        size = 0
        count = 0
        while index < len(lines) and (count == 0 or size + len(lines[index]) + 1 <= max_bytes):
            size += len(lines[index]) + 1
            offset += len(lines[index]) + 1
            row += 1
            index += 1
            count += 1
        end_offset = min(offset - 1, piece.end)
        windows.append(
            replace(piece, start=start_offset, end=end_offset, start_row=start_row, end_row=row - 1)
        )
        if index < len(lines) and count > LINE_WINDOW_OVERLAP:
            # Küçük bir örtüşme: pencere sınırındaki bir ifade iki parçada da okunabilsin.
            back = LINE_WINDOW_OVERLAP
            for _ in range(back):
                index -= 1
                offset -= len(lines[index]) + 1
                row -= 1
    return windows


def _text(node: Node | None, data: bytes) -> str:
    if node is None:
        return ""
    return data[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


# ----------------------------------------------------------------- markdown


def _chunk_markdown(path: str, text: str, config: ChunkerConfig) -> list[ChunkRecord]:
    sections = _sections(text)
    records: list[ChunkRecord] = []
    buffer: list[tuple[int, int, str]] = []

    def header_for(heading: str, part: int = 1, parts: int = 1) -> str:
        lines = [f"# file: {path}"]
        if heading:
            suffix = f" [part {part}/{parts}]" if parts > 1 else ""
            lines.append(f"# section: {heading}{suffix}")
        return "\n".join(lines)

    def emit(start: int, end: int, heading: str, part: int = 1, parts: int = 1) -> None:
        body = text[start:end]
        if not body.strip():
            return
        title = heading.split(" > ")[-1] if heading else ""
        records.append(
            ChunkRecord(
                ordinal=len(records),
                text=body,
                start_line=text.count("\n", 0, start) + 1,
                end_line=text.count("\n", 0, max(end - 1, start)) + 1,
                symbol=title,
                parent_symbol=" > ".join(heading.split(" > ")[:-1]),
                kind="section",
                header=header_for(heading, part, parts),
                symbols=(title,) if title else (),
            )
        )

    def flush() -> None:
        if buffer:
            emit(buffer[0][0], buffer[-1][1], buffer[0][2])
            buffer.clear()

    for start, end, heading in sections:
        size = _byte_len(text[start:end])
        if size > config.max_bytes:
            flush()
            windows = _paragraph_windows(text, start, end, config.max_bytes)
            for index, (window_start, window_end) in enumerate(windows, start=1):
                emit(window_start, window_end, heading, index, len(windows))
            continue
        if buffer and _byte_len(text[buffer[0][0] : end]) > config.max_bytes:
            flush()
        buffer.append((start, end, heading))
        if _byte_len(text[buffer[0][0] : end]) >= config.min_bytes:
            flush()
    flush()
    return records


def _sections(text: str) -> list[tuple[int, int, str]]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [(0, len(text), "")]
    sections: list[tuple[int, int, str]] = []
    if matches[0].start() > 0:
        sections.append((0, matches[0].start(), ""))
    path: list[str] = []
    for index, match in enumerate(matches):
        depth = len(match.group(1))
        del path[depth - 1 :]
        path.append(match.group(2).strip())
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.start(), end, " > ".join(path)))
    return sections


# -------------------------------------------------------------------- plain


def _chunk_plain(path: str, text: str, config: ChunkerConfig) -> list[ChunkRecord]:
    header = f"# file: {path}"
    records: list[ChunkRecord] = []
    windows = _paragraph_windows(text, 0, len(text), config.max_bytes)
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if (
            merged
            and _byte_len(text[merged[-1][0] : end]) <= config.max_bytes
            and _byte_len(text[merged[-1][0] : merged[-1][1]]) < config.min_bytes
        ):
            merged[-1] = (merged[-1][0], end)
            continue
        merged.append((start, end))
    for start, end in merged:
        body = text[start:end]
        if not body.strip():
            continue
        records.append(
            ChunkRecord(
                ordinal=len(records),
                text=body,
                start_line=text.count("\n", 0, start) + 1,
                end_line=text.count("\n", 0, max(end - 1, start)) + 1,
                symbol="",
                parent_symbol="",
                kind="block",
                header=header,
            )
        )
    return records


def _paragraph_windows(text: str, start: int, end: int, max_bytes: int) -> list[tuple[int, int]]:
    """Boş satırlarda böl, MAX'a kadar paketle; tek paragraf MAX'ı aşarsa satırlara in."""
    boundaries = [start, *(m.end() for m in _BLANK_LINES.finditer(text, start, end)), end]
    windows: list[tuple[int, int]] = []
    window_start = start
    for cursor in boundaries[1:]:
        if _byte_len(text[window_start:cursor]) > max_bytes:
            previous = windows[-1][1] if windows else window_start
            if previous > window_start:
                windows.append((window_start, previous))
                window_start = previous
            if _byte_len(text[window_start:cursor]) > max_bytes:
                windows.extend(_line_slices(text, window_start, cursor, max_bytes))
                window_start = cursor
                continue
        if cursor == end:
            if cursor > window_start:
                windows.append((window_start, cursor))
        else:
            # Pencereyi açık tut; bir sonraki paragraf da sığabilir.
            if windows and windows[-1][1] == window_start:
                continue
            windows.append((window_start, cursor))
            window_start = cursor
    return _coalesce(text, windows, max_bytes)


def _coalesce(text: str, windows: list[tuple[int, int]], max_bytes: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if merged and _byte_len(text[merged[-1][0] : end]) <= max_bytes:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _line_slices(text: str, start: int, end: int, max_bytes: int) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        limit = cursor
        size = 0
        while limit < end:
            newline = text.find("\n", limit, end)
            line_end = end if newline == -1 else newline + 1
            line_size = _byte_len(text[limit:line_end])
            if size and size + line_size > max_bytes:
                break
            size += line_size
            limit = line_end
            if line_size > max_bytes:
                break
        if limit == cursor:
            limit = min(cursor + max_bytes, end)
        slices.append((cursor, limit))
        cursor = limit
    return slices


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def chunk_many(
    files: Iterable[tuple[str, str]], config: ChunkerConfig | None = None
) -> dict[str, list[ChunkRecord]]:
    return {path: chunk_file(path, text, config) for path, text in files}
