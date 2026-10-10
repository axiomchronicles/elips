import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/vault")({
  head: () => ({
    meta: [
      { title: "Vault — ELIPS Python API Docs" },
      {
        name: "description",
        content:
          "Reference for the low-level native ELIPS Vault reached via collection.native: place, seek, scan, fetch, and erase. The supported high-level surface is Collection.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="PYTHON API · LOW-LEVEL"
      title="Vault"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "place", label: "place()" },
        { id: "place-many", label: "place_many()" },
        { id: "seek", label: "seek()" },
        { id: "scan", label: "scan()" },
        { id: "fetch", label: "fetch()" },
        { id: "erase", label: "erase()" },
        { id: "info", label: "info() & Properties" },
        { id: "result-shape", label: "Result Shape" },
        { id: "thread-safety", label: "Thread Safety" },
        { id: "examples", label: "Full Examples" },
      ]}
    >
      <h2 id="overview">Overview</h2>
      <p>
        A native <code>Vault</code> (<code>elips.native.Vault</code>) is the
        low-level escape hatch beneath a{" "}
        <a href="/docs/python/arena">Collection</a>. The{" "}
        <strong>supported</strong> high-level surface is the Collection —{" "}
        <code>add</code>, <code>query</code>, <code>get</code>, and{" "}
        <code>delete</code>. Reach for the native vault only when you need a
        capability the collection does not wrap.
      </p>
      <p>
        You never construct a <code>Vault</code> directly. Reach one via{" "}
        <code>collection.native</code>, or from the native{" "}
        <a href="/docs/python/database">Database</a> via{" "}
        <code>client.native.vault("name")</code>.
      </p>
      <CodeBlock lang="python">{`import elips

client = elips.Client(dimension=768, metric="cosine")
products = client.get_or_create_collection("products")

# Drop to the native vault for low-level verbs
vault = products.native          # elips.native.Vault
# ...or straight from the native Database:
vault = client.native.vault("products")
`}</CodeBlock>
      <p>
        The native verbs are <code>place</code>, <code>seek</code>,{" "}
        <code>fetch</code>, <code>erase</code>, and <code>scan</code>. These are
        the raw runtime operations; the Collection's <code>add</code> /{" "}
        <code>query</code> / <code>get</code> / <code>delete</code> are the
        supported, typed wrappers over them. Text and hybrid retrieval live on
        the Collection (<code>query(query_texts=...)</code>), not here.
      </p>
      <h2 id="place">
        <code>Vault.place()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.place(
    vector:   list[float] | np.ndarray,
    data:     dict         = {},
    id:       str | None   = None,
    document: str | None   = None,
    chunk:    dict | None  = None,
    lineage:  dict | None  = None,
) -> str
`}</CodeBlock>
      <p>
        Insert a single record and return its id (your supplied <code>id</code>{" "}
        or an auto-generated one). Re-using an existing id overwrites that record
        (upsert).
      </p>
      <ul>
        <li>
          <strong>
            <code>vector</code>
          </strong>{" "}
          — embedding; length must match the database dimension or{" "}
          <code>DimensionMismatch</code> is raised.
        </li>
        <li>
          <strong>
            <code>data</code>
          </strong>{" "}
          — JSON-serialisable metadata used in filters.
        </li>
        <li>
          <strong>
            <code>id</code>
          </strong>{" "}
          — custom string id; auto-generated if omitted.
        </li>
        <li>
          <strong>
            <code>document</code>
          </strong>{" "}
          — original text the vector was derived from; returned in results.
        </li>
        <li>
          <strong>
            <code>chunk</code>
          </strong>{" "}
          /{" "}
          <strong>
            <code>lineage</code>
          </strong>{" "}
          — optional chunk and embedding provenance.
        </li>
      </ul>
      <CodeBlock lang="python">{`rid = vault.place(embedding, data={"sku": "BOOT-42", "price": 129.99})

# Deterministic id (upsert on repeat)
vault.place(
    embedding,
    id="product:BOOT-42",
    data={"sku": "BOOT-42", "price": 139.99},
    document="Leather hiking boot, size 42",
    lineage={"provider": "openai", "model": "text-embedding-3-small"},
)
`}</CodeBlock>
      <h2 id="place-many">
        <code>Vault.place_many()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.place_many(records: list[dict]) -> None
`}</CodeBlock>
      <p>
        Bulk-insert records in a single implicit transaction. Each record is a
        dict with a <code>vector</code> plus optional <code>id</code>,{" "}
        <code>data</code>, <code>document</code>, <code>chunk</code>, and{" "}
        <code>lineage</code>. Faster than looping <code>place()</code> because it
        amortises graph-index and WAL costs.
      </p>
      <CodeBlock lang="python">{`records = [
    {"vector": embed(t), "data": {"title": t}, "id": f"doc:{i}"}
    for i, t in enumerate(texts)
]
vault.place_many(records)
`}</CodeBlock>

      <h2 id="seek">
        <code>Vault.seek()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.seek(
    vector:    list[float] | np.ndarray,
    top:       int,
    where:     dict | None  = None,
    threshold: float | None = None,
) -> list[Hit]
`}</CodeBlock>
      <p>
        Approximate nearest-neighbour search over the HNSW graph, returning up to{" "}
        <code>top</code> hits ordered by ascending distance. <code>where</code>{" "}
        is an optional metadata filter; <code>threshold</code> drops hits beyond
        a maximum distance.
      </p>
      <CodeBlock lang="python">{`hits = vault.seek(query_vec, top=10)
for h in hits:
    print(h.key, h.distance, h.meta)

# With a metadata filter and distance cutoff
hits = vault.seek(query_vec, top=5, where={"category": "shoes"}, threshold=0.35)
`}</CodeBlock>
      <h2 id="scan">
        <code>Vault.scan()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.scan(
    where:  dict | None = None,
    offset: int = 0,
    limit:  int = -1,
) -> list[Row]
`}</CodeBlock>
      <p>
        Full sequential scan with an optional <code>where</code> filter. Returns{" "}
        <code>Row</code> objects (no distance). Use for export, re-indexing, or
        auditing — not for latency-critical paths. <code>limit=-1</code> returns
        everything.
      </p>
      <CodeBlock lang="python">{`all_records = vault.scan()
page1 = vault.scan(offset=0,   limit=100)
page2 = vault.scan(offset=100, limit=100)
cheap = vault.scan(where={"category": "shoes"})
`}</CodeBlock>

      <h2 id="fetch">
        <code>Vault.fetch()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.fetch(id: str) -> Row | None
`}</CodeBlock>
      <p>
        Retrieve a single record by id as a <code>Row</code> (with{" "}
        <code>.key</code>, <code>.meta</code>, <code>.text</code>,{" "}
        <code>.vector</code>), or <code>None</code> if no such record exists.
      </p>
      <CodeBlock lang="python">{`row = vault.fetch("product:BOOT-42")
if row:
    print(row.meta["price"])
    print(row.vector[:5])
`}</CodeBlock>

      <h2 id="erase">
        <code>Vault.erase()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.erase(id: str) -> bool
`}</CodeBlock>
      <p>
        Delete the record with the given id. Returns <code>True</code> if it
        existed, <code>False</code> otherwise. Erases are tombstones: the record
        is immediately invisible to searches and scans, and the storage is
        reclaimed on the next <code>vacuum</code> / <code>compact</code>.
      </p>
      <CodeBlock lang="python">{`deleted = vault.erase("product:BOOT-42")
print(deleted)   # True if it existed
`}</CodeBlock>
      <h2 id="info">
        <code>info()</code> &amp; Properties
      </h2>
      <h3>
        <code>{"Vault.info() -> VaultInfo"}</code>
      </h3>
      <p>Returns a snapshot with:</p>
      <ul>
        <li>
          <code>.count: int</code> — live (non-erased) record count.
        </li>
        <li>
          <code>.dimension: int</code> — vector dimension.
        </li>
        <li>
          <code>.metric: str</code> — distance metric.
        </li>
      </ul>
      <CodeBlock lang="python">{`info = vault.info()
print(f"{info.count} records, {info.dimension}d, metric={info.metric}")
`}</CodeBlock>
      <h3>
        <code>Vault.name: str</code>
      </h3>
      <p>The name this vault was registered under.</p>
      <h3>
        <code>Vault.count: int</code>
      </h3>
      <p>
        Shorthand for <code>vault.info().count</code> — live records only.
      </p>
      <h3>
        <code>Vault.pending_removals: int</code>
      </h3>
      <p>
        Count of erased-but-not-yet-purged records. A high value relative to{" "}
        <code>count</code> is a sign to vacuum (via the owning collection or
        client).
      </p>
      <h3>
        <code>Vault.read_only: bool</code>
      </h3>
      <p>
        <code>True</code> when the vault refuses writes (database opened
        read-only, or the collection frozen).
      </p>
      <h3>
        <code>Vault.sealed: bool</code>
      </h3>
      <p>
        <code>True</code> once the owning database is closed; writes then raise.
      </p>
      <h2 id="result-shape">Result Shape</h2>
      <p>
        <code>seek()</code> returns <code>Hit</code> objects; <code>scan()</code>{" "}
        and <code>fetch()</code> return <code>Row</code> objects. Field names are{" "}
        <code>.key</code> and <code>.meta</code> — not <code>.id</code> or{" "}
        <code>.data</code>.
      </p>
      <ul>
        <li>
          <code>.key: str</code> — record identifier.
        </li>
        <li>
          <code>.distance: float</code> — distance from the query (
          <code>Hit</code> only; lower is closer for cosine / euclidean).
        </li>
        <li>
          <code>.meta: dict</code> — stored metadata.
        </li>
        <li>
          <code>.text: str | None</code> — document text, when a{" "}
          <code>document</code> was stored.
        </li>
        <li>
          <code>.vector</code> — stored embedding, when hydrated.
        </li>
      </ul>
      <CodeBlock lang="python">{`hits = vault.seek(query_vec, top=5)
for h in hits:
    print(f"key={h.key}  dist={h.distance:.4f}  meta={h.meta}")
    if h.text:
        print(f"  text: {h.text[:80]}")
`}</CodeBlock>

      <h2 id="thread-safety">Thread Safety</h2>
      <p>
        Each native <code>Vault</code> uses a reader-writer lock. Reads —{" "}
        <code>seek()</code>, <code>scan()</code>, <code>fetch()</code>,{" "}
        <code>info()</code>, <code>count</code> — take a shared lock and run
        concurrently. Writes — <code>place()</code>, <code>place_many()</code>,{" "}
        <code>erase()</code>, transaction commits — take an exclusive lock. The
        model favours read-heavy workloads; batch writes via{" "}
        <code>place_many()</code> to amortise lock contention.
      </p>
      <h2 id="examples">Full Examples</h2>
      <h3>Low-level ingestion through the native vault</h3>
      <CodeBlock lang="python">{`import elips, itertools

client = elips.PersistentClient("/data/docs", dimension=1536, metric="cosine")
vault = client.get_or_create_collection("docs").native

def ingest_chunks(chunks: list[dict], batch_size: int = 500):
    it = iter(chunks)
    for batch in iter(lambda: list(itertools.islice(it, batch_size)), []):
        records = [
            {
                "vector":   c["vector"],
                "data":     {"source": c["file"], "page": c["page"]},
                "document": c["text"],
            }
            for c in batch
        ]
        vault.place_many(records)
    print(f"Total: {vault.count} records")
`}</CodeBlock>

      <h3>Semantic search with a filter</h3>
      <CodeBlock lang="python">{`def search_products(vec, category: str, n: int = 10):
    return vault.seek(
        vec,
        top=n,
        where={"category": category, "in_stock": True},
        threshold=0.5,
    )

hits = search_products(query_vec, "footwear")
for h in hits:
    print(h.meta["sku"], h.distance)
`}</CodeBlock>
    </DocsShell>
  );
}
