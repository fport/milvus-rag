from milvus_rag.index.chunk import (
    ChunkerConfig,
    build_indexed_text,
    chunk_file,
    split_identifiers,
)

TS = """import { db } from "@/db";
import { redis } from "@/shared/lib/redis";

/** Resolves the user from the session. */
export async function withSession(c: Context, next: Next) {
  const session = await auth.api.getSession({ headers: c.req.raw.headers });
  c.set("session", session);
  await next();
}

export const requireAuth = createMiddleware(async (c, next) => {
  if (!c.get("session")) throw new UnauthorizedError();
  await next();
});

export class SessionCache {
  private readonly ttl = 60;

  constructor(private readonly client: Redis) {}

  async get(key: string): Promise<string | null> {
    return this.client.get(key);
  }

  async set(key: string, value: string): Promise<void> {
    await this.client.set(key, value, "EX", this.ttl);
  }

  async del(key: string): Promise<void> {
    await this.client.del(key);
  }

  async has(key: string): Promise<boolean> {
    return (await this.client.exists(key)) === 1;
  }
}

type Handler = (c: Context) => Promise<Response>;
"""


def test_typescript_units_have_symbols_and_lines():
    records = chunk_file("src/middlewares/auth.ts", TS, ChunkerConfig(max_bytes=400, min_bytes=80))
    symbols = {record.symbol for record in records}
    assert {"withSession", "requireAuth"} <= symbols
    assert all(record.start_line <= record.end_line for record in records)
    assert all(len(record.text.encode()) <= 400 or record.kind for record in records)
    # The class did not fit, so it was split into methods; each carries the class as parent.
    methods = [record for record in records if record.parent_symbol == "SessionCache"]
    assert {record.symbol for record in methods} >= {"get", "set"}
    assert all("file: src/middlewares/auth.ts" in record.header for record in records)
    first = next(record for record in records if record.symbol == "withSession")
    assert "imports:" in first.header and "@/db" in first.header


def test_small_pieces_merge_and_doc_comment_sticks_to_function():
    records = chunk_file("a.ts", TS, ChunkerConfig(max_bytes=2000, min_bytes=200))
    with_session = next(record for record in records if record.symbol == "withSession")
    assert "Resolves the user from the session" in with_session.text
    assert with_session.text.startswith("import")  # the import block sticks to the first chunk


def test_oversized_function_is_windowed_but_keeps_symbol():
    body = "\n".join(f"  const v{i} = compute({i});" for i in range(200))
    text = f"export function huge() {{\n{body}\n}}\n"
    records = chunk_file("big.ts", text, ChunkerConfig(max_bytes=1500, min_bytes=200))
    assert len(records) > 1
    assert all(record.symbol == "huge" for record in records)
    assert all(len(record.text.encode()) <= 1500 + 80 for record in records)
    assert "[part 1/" in records[0].header
    assert records[0].start_line == 1 and records[-1].end_line == 202


def test_python_class_and_decorators():
    source = (
        "import os\n\nCONST = 5\n\n@dataclass\nclass Foo:\n    a: int = 1\n\n"
        "    def bar(self, x):\n        return x + self.a\n\n"
        "    def baz(self):\n        return 1\n\ndef top(a, b):\n    return a + b\n"
    )
    records = chunk_file("t.py", source, ChunkerConfig(max_bytes=80, min_bytes=30))
    symbols = [(record.symbol, record.parent_symbol) for record in records]
    assert ("bar", "Foo") in symbols and ("baz", "Foo") in symbols
    assert any(record.symbol == "top" for record in records)
    assert any(record.header.startswith("# file:") for record in records)


def test_csharp_file_scoped_namespace_members():
    source = (
        "using System;\nnamespace Foo.Bar;\n\npublic class Svc\n{\n"
        "    private readonly int _a;\n    public Svc(int a) { _a = a; }\n"
        "    public int Add(int x) => x + _a;\n    public int Sub(int x) { return x - _a; }\n}\n"
    )
    records = chunk_file("Svc.cs", source, ChunkerConfig(max_bytes=80, min_bytes=20))
    symbols = {record.symbol for record in records}
    assert {"Add", "Sub"} <= symbols
    assert any(record.parent_symbol.endswith("Svc") for record in records)


def test_markdown_sections_keep_heading_path():
    source = (
        "# Guide\n\nintro\n\n## Setup\n\nrun it\n\n### Docker\n\ncompose up\n\n## Usage\n\nask\n"
    )
    records = chunk_file("README.md", source, ChunkerConfig(max_bytes=60, min_bytes=10))
    headers = [record.header for record in records]
    assert any("Guide > Setup > Docker" in header for header in headers)
    assert all(record.kind == "section" for record in records)
    assert records[0].start_line == 1


def test_plain_text_and_json_fall_back_to_windows():
    source = "\n\n".join(f"paragraph {i} " + "x" * 50 for i in range(20))
    records = chunk_file("notes.txt", source, ChunkerConfig(max_bytes=300, min_bytes=100))
    assert records and all(len(record.text.encode()) <= 300 for record in records)
    assert "".join(record.text for record in records).replace("\n", "") == source.replace("\n", "")


def test_split_identifiers():
    tokens = split_identifiers("handleAuthCallback MILVUS_HOST HTTPServer getUserByID").split()
    assert {"handle", "auth", "callback", "milvus", "host", "http", "server", "user"} <= set(tokens)


def test_indexed_text_layers():
    records = chunk_file("a.ts", TS, ChunkerConfig(max_bytes=400, min_bytes=80))
    record = next(record for record in records if record.symbol == "withSession")
    indexed = build_indexed_text(record, context="Oturumu okur.")
    assert indexed.startswith("// file: a.ts")
    assert "Oturumu okur." in indexed
    assert record.text in indexed
    assert "session" in indexed.split("\n")[-1]


def test_empty_file_yields_nothing():
    assert chunk_file("x.ts", "   \n", ChunkerConfig()) == []


def test_razor_code_block_members_become_units():
    source = (
        '@page "/orders"\n@inject OrderService Svc\n<h1>Orders</h1>\n@code {\n'
        "    private List<Order> orders = new();\n"
        "    protected override async Task OnInitializedAsync() { orders = await Svc.All(); }\n"
        "    void Save() { Svc.Save(orders); }\n}\n"
    )
    records = chunk_file("Pages/Orders.razor", source, ChunkerConfig(max_bytes=120, min_bytes=20))
    symbols = {record.symbol for record in records}
    assert {"OnInitializedAsync", "Save"} <= symbols
    assert records[0].header.startswith("// file: Pages/Orders.razor")
