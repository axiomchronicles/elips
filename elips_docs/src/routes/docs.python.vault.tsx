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
          "Complete reference for the ELIPS Python Vault class: inserting, searching, scanning, erasing, and managing vectors within a named namespace.",
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
        { id: "place-document", label: "place_document()" },
        { id: "place-many", label: "place_many()" },
        { id: "seek", label: "seek()" },
        { id: "seek-text", label: "seek_text()" },
        { id: "seek-hybrid", label: "seek_hybrid()" },
        { id: "explain-seek", label: "explain_seek()" },
        { id: "scan", label: "scan()" },
        { id: "fetch", label: "fetch()" },
        { id: "erase", label: "erase()" },
        { id: "info", label: "info() & Properties" },
        { id: "result-shape", label: "Result Shape" },
        { id: "thread-safety", label: "Thread Safety" },
        { id: "examples", label: "Full Examples" },
      ]}
    >
      {/* ── Overview ───────────────────────────────────────────── */}
      <h2 id="overview">Overview</h2>
      <p>
        A <code>Vault</code> is a named namespace within a{" "}
        <a href="/docs/python/database">Database</a>. All vectors stored in a
        vault share the same dimension and distance metric (inherited from the
        database). You can have as many vaults as you like — they are cheap,
        lazy-initialised, and each carries its own HNSW graph index and record
        store.
      </p>
      <p>
        You never construct a <code>Vault</code> directly; obtain one via{" "}
        <code>db.vault("name")</code>.
      </p>
      <CodeBlock lang="python">{`db = elips.open("/data/shop.elips", dimension=768, metric="cosine")
products = db.vault("products")  # created on first access
`}</CodeBlock>

      {/* ── place() ────────────────────────────────────────────── */}
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

      <h3>Parameters</h3>
      <ul>
        <li>
          <strong>
            <code>vector</code>
          </strong>{" "}
          — The embedding as a list of floats or a 1-D NumPy array. Length must
          match the database dimension; mismatches raise{" "}
          <code>DimensionMismatch</code>.
        </li>
        <li>
          <strong>
            <code>data</code>
          </strong>{" "}
          — Arbitrary JSON-serialisable metadata attached to the record. Used
          in filter expressions inside <code>seek()</code> and EQL queries.
          Defaults to an empty dict.
        </li>
        <li>
          <strong>
            <code>id</code>
          </strong>{" "}
          — Custom string identifier. If <code>None</code> (default), ELIPS
          generates a UUID v4. If you supply an ID that already exists in the
          vault, the record is updated in-place (upsert semantics).
        </li>
        <li>
          <strong>
            <code>document</code>
          </strong>{" "}
          — Original text from which the vector was derived. Stored as a{" "}
          <code>DocumentAttachment</code> and returned in seek results.
          Optional.
        </li>
        <li>
          <strong>
            <code>chunk</code>
          </strong>{" "}
          — Chunk provenance information (e.g., source file, byte offset,
          page). Stored as <code>ChunkInfo</code>. Optional.
        </li>
        <li>
          <strong>
            <code>lineage</code>
          </strong>{" "}
          — Embedding provenance: model name, provider, revision. Stored as{" "}
          <code>EmbeddingLineage</code>. Optional.
        </li>
      </ul>

      <h3>Return value</h3>
      <p>
        The string ID of the placed record (either your custom ID or the
        auto-generated UUID).
      </p>

      <CodeBlock lang="python">{`# Basic placement
rid = products.place(embedding, data={"sku": "BOOT-42", "price": 129.99})
print(rid)  # 'f47ac10b-58cc-4372-a567-0e02b2c3d479'

# Deterministic ID (upsert if same ID)
products.place(
    embedding,
    id="product:BOOT-42",
    data={"sku": "BOOT-42", "price": 139.99},
    document="Leather hiking boot, size 42",
    lineage={"provider": "openai", "model": "text-embedding-3-small"},
)
`}</CodeBlock>

      {/* ── place_document() ───────────────────────────────────── */}
      <h2 id="place-document">
        <code>Vault.place_document()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.place_document(
    text:    str,
    data:    dict        = {},
    id:      str | None  = None,
    chunk:   dict | None = None,
    lineage: dict | None = None,
) -> str
`}</CodeBlock>
      <p>
        A convenience wrapper that calls the database's configured text embedder
        to produce a vector from <code>text</code>, then calls{" "}
        <code>place()</code>. The <code>document</code> field is automatically
        set to the raw <code>text</code>.
      </p>
      <p>
        Requires that the database was opened with an{" "}
        <code>embedder</code> / <code>text_embedder</code> configured in{" "}
        <code>Config</code>. Raises <code>RuntimeError</code> if no embedder is
        attached.
      </p>

      <CodeBlock lang="python">{`# Database must have an embedder configured
db = elips.open(
    "/data/docs.elips",
    dimension=1536,
    metric="cosine",
    embedder=my_openai_embed_fn,
)
docs = db.vault("docs")

rid = docs.place_document(
    "ELIPS is an embedded C++23 vector database.",
    data={"source": "readme.md", "section": "intro"},
)
`}</CodeBlock>

      {/* ── place_many() ───────────────────────────────────────── */}
      <h2 id="place-many">
        <code>Vault.place_many()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.place_many(records: list[dict]) -> None
`}</CodeBlock>
      <p>
        Bulk-inserts a list of records in a single operation. Each record is a
        dict that may contain any combination of the following keys (all
        optional except one of <code>vector</code> or <code>text</code>):
      </p>

      <ul>
        <li>
          <code>vector</code> — Embedding (list or ndarray). Required if{" "}
          <code>text</code> is absent.
        </li>
        <li>
          <code>text</code> — Raw text. ELIPS will embed it automatically if an
          embedder is configured. Required if <code>vector</code> is absent.
        </li>
        <li>
          <code>id</code> — Optional string ID (auto-generated if omitted).
        </li>
        <li>
          <code>data</code> — Metadata dict (defaults to{" "}
          <code>{"{}"}</code>).
        </li>
        <li>
          <code>document</code> — Original document text.
        </li>
        <li>
          <code>chunk</code> — Chunk info dict.
        </li>
        <li>
          <code>lineage</code> — Embedding lineage dict.
        </li>
      </ul>

      <p>
        <code>place_many()</code> writes all records within a single implicit
        transaction for efficiency: either all succeed or all fail. It is
        significantly faster than calling <code>place()</code> in a loop
        because it amortises graph-index insertion and WAL write costs.
      </p>

      <CodeBlock lang="python">{`records = [
    {"vector": embed(t), "data": {"title": t}, "id": f"doc:{i}"}
    for i, t in enumerate(texts)
]
vault.place_many(records)

# Mixing text and vector records (requires embedder)
mixed = [
    {"text": "hello world", "data": {"lang": "en"}},
    {"vector": [0.1] * 768, "data": {"source": "manual"}},
]
vault.place_many(mixed)
`}</CodeBlock>

      <h3>Performance note</h3>
      <p>
        For large ingestion jobs (&gt; 100 k vectors), prefer{" "}
        <code>place_many()</code> in batches of 1 000–10 000 records over
        individual <code>place()</code> calls. The HNSW graph is updated
        incrementally, so very large single batches may cause a temporary
        increase in memory usage.
      </p>

      {/* ── seek() ─────────────────────────────────────────────── */}
      <h2 id="seek">
        <code>Vault.seek()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.seek(
    vector:    list[float] | np.ndarray,
    top:       int,
    where:     Filter = Filter(),
    threshold: float | None = None,
) -> list[Result]
`}</CodeBlock>
      <p>
        Performs an approximate nearest-neighbour (ANN) search using the HNSW
        graph index, returning up to <code>top</code> results ordered by
        ascending distance to <code>vector</code>.
      </p>

      <h3>Parameters</h3>
      <ul>
        <li>
          <strong>
            <code>vector</code>
          </strong>{" "}
          — Query embedding. Must match the database dimension.
        </li>
        <li>
          <strong>
            <code>top</code>
          </strong>{" "}
          — Maximum number of results to return. Actual results may be fewer
          if the vault contains fewer records or if <code>threshold</code>{" "}
          eliminates candidates.
        </li>
        <li>
          <strong>
            <code>where</code>
          </strong>{" "}
          — A <code>Filter</code> expression for post-processing candidate
          results by metadata. Example:{" "}
          <code>{'Filter(category="shoes", price__lt=200)'}</code>. Defaults
          to no filter (<code>Filter()</code>).
        </li>
        <li>
          <strong>
            <code>threshold</code>
          </strong>{" "}
          — Maximum distance cutoff. Results with distance strictly greater
          than <code>threshold</code> are excluded. Disabled when{" "}
          <code>None</code>.
        </li>
      </ul>

      <h3>Return value</h3>
      <p>
        A list of <a href="#result-shape"><code>Result</code></a> objects,
        sorted by ascending distance.
      </p>

      <CodeBlock lang="python">{`from elips import Filter

results = vault.seek(query_vec, top=10)
for r in results:
    print(r.id, r.distance, r.data)

# With metadata filter
results = vault.seek(
    query_vec,
    top=5,
    where=Filter(category="shoes", in_stock=True),
    threshold=0.35,
)
`}</CodeBlock>

      {/* ── seek_text() ────────────────────────────────────────── */}
      <h2 id="seek-text">
        <code>Vault.seek_text()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.seek_text(
    text:      str,
    top:       int,
    where:     Filter      = Filter(),
    threshold: float | None = None,
) -> list[Result]
`}</CodeBlock>
      <p>
        Embeds <code>text</code> using the configured embedder, then runs a
        vector search identical to <code>seek()</code>. Requires an embedder.
      </p>

      <CodeBlock lang="python">{`results = vault.seek_text("comfortable running shoes", top=10)
`}</CodeBlock>

      {/* ── seek_hybrid() ──────────────────────────────────────── */}
      <h2 id="seek-hybrid">
        <code>Vault.seek_hybrid()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.seek_hybrid(
    vector:         list[float] | np.ndarray,
    text:           str,
    top:            int,
    where:          Filter      = Filter(),
    threshold:      float | None = None,
    lexical_weight: float        = 0.25,
) -> list[Result]
`}</CodeBlock>
      <p>
        Combines dense vector similarity with a BM25 lexical term-frequency
        score. The final score is a weighted sum:
      </p>
      <CodeBlock lang="python">{`final_score = (1 - lexical_weight) * vector_score + lexical_weight * bm25_score
`}</CodeBlock>

      <h3>Parameters</h3>
      <ul>
        <li>
          <strong>
            <code>vector</code>
          </strong>{" "}
          — Dense query embedding.
        </li>
        <li>
          <strong>
            <code>text</code>
          </strong>{" "}
          — Lexical query string for BM25 scoring.
        </li>
        <li>
          <strong>
            <code>lexical_weight</code>
          </strong>{" "}
          — Weight of the lexical component in the final score. Must be in{" "}
          <code>[0.0, 1.0]</code>. A value of <code>0.0</code> degrades to
          pure vector search; <code>1.0</code> degrades to pure BM25. Default{" "}
          <code>0.25</code>.
        </li>
      </ul>

      <p>
        Hybrid search requires that records were placed with a{" "}
        <code>document</code> field. Records without documents are still
        included but receive a BM25 score of 0.
      </p>

      <CodeBlock lang="python">{`results = vault.seek_hybrid(
    vector=dense_embedding,
    text="waterproof hiking boot",
    top=10,
    lexical_weight=0.3,
)
`}</CodeBlock>

      {/* ── explain_seek() ─────────────────────────────────────── */}
      <h2 id="explain-seek">
        <code>Vault.explain_seek()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.explain_seek(
    vector:             list[float] | np.ndarray,
    top:                int,
    where:              Filter      = Filter(),
    threshold:          float | None = None,
    has_text_component: bool         = False,
) -> QueryPlan
`}</CodeBlock>
      <p>
        Returns a <code>QueryPlan</code> describing the execution plan for the
        given seek, without actually running the search. Use this during
        development to verify that filters are being applied at the expected
        stage and to estimate the number of graph hops.
      </p>

      <CodeBlock lang="python">{`plan = vault.explain_seek(query_vec, top=10, where=Filter(category="shoes"))
print(plan.strategy)         # 'hnsw_with_post_filter'
print(plan.estimated_hops)   # 42
print(plan.filter_stage)     # 'post'
`}</CodeBlock>

      {/* ── scan() ─────────────────────────────────────────────── */}
      <h2 id="scan">
        <code>Vault.scan()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.scan(
    where:  Filter = Filter(),
    offset: int    = 0,
    limit:  int    = -1,
) -> list[dict]
`}</CodeBlock>
      <p>
        Performs a full sequential scan over all records in the vault, applying
        the optional <code>where</code> filter, and returns a page of plain
        dicts (not <code>Result</code> objects — no distance field). Use for
        data export, re-indexing, or auditing. Not intended for latency-critical
        paths.
      </p>

      <h3>Parameters</h3>
      <ul>
        <li>
          <strong>
            <code>where</code>
          </strong>{" "}
          — Metadata filter applied during the scan.
        </li>
        <li>
          <strong>
            <code>offset</code>
          </strong>{" "}
          — Number of records to skip (for pagination). Defaults to{" "}
          <code>0</code>.
        </li>
        <li>
          <strong>
            <code>limit</code>
          </strong>{" "}
          — Maximum number of records to return. <code>-1</code> (default)
          means no limit — all matching records are returned.
        </li>
      </ul>

      <CodeBlock lang="python">{`# Export all records
all_records = vault.scan()

# Paginate
page1 = vault.scan(offset=0,   limit=100)
page2 = vault.scan(offset=100, limit=100)

# Filter-only scan
cheap = vault.scan(where=Filter(price__lt=50))
`}</CodeBlock>

      {/* ── fetch() ────────────────────────────────────────────── */}
      <h2 id="fetch">
        <code>Vault.fetch()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.fetch(id: str) -> dict | None
`}</CodeBlock>
      <p>
        Retrieves a single record by ID. Returns a dict with all stored fields
        (including <code>vector</code>, <code>data</code>,{" "}
        <code>document</code>, <code>chunk</code>, <code>lineage</code>) or{" "}
        <code>None</code> if no record with that ID exists.
      </p>

      <CodeBlock lang="python">{`rec = vault.fetch("product:BOOT-42")
if rec:
    print(rec["data"]["price"])
    print(rec["vector"][:5])
`}</CodeBlock>

      {/* ── erase() ────────────────────────────────────────────── */}
      <h2 id="erase">
        <code>Vault.erase()</code>
      </h2>
      <CodeBlock lang="python">{`Vault.erase(id: str) -> bool
`}</CodeBlock>
      <p>
        Marks the record with the given ID as deleted. Returns <code>True</code>{" "}
        if the record existed and was erased, <code>False</code> if the ID was
        not found.
      </p>
      <p>
        Erased records are immediately invisible to future searches and scans.
        The underlying storage is not reclaimed until a{" "}
        <code>vacuum()</code> or <code>compact()</code> is performed. The
        accumulation of erased records is tracked by the{" "}
        <code>pending_removals</code> property; once this exceeds the vault's{" "}
        <code>compaction_ratio</code> threshold, an automatic graph rebuild is
        triggered on the next write.
      </p>

      <CodeBlock lang="python">{`deleted = vault.erase("product:BOOT-42")
print(deleted)  # True if it existed
`}</CodeBlock>

      {/* ── info() & Properties ────────────────────────────────── */}
      <h2 id="info">
        <code>info()</code> &amp; Properties
      </h2>

      <h3>
        <code>{"Vault.info() -> VaultInfo"}</code>
      </h3>
      <p>
        Returns a <code>VaultInfo</code> snapshot with:
      </p>
      <ul>
        <li>
          <code>.count: int</code> — Number of live (non-erased) records.
        </li>
        <li>
          <code>.dimension: int</code> — Vector dimension (same as the
          database).
        </li>
        <li>
          <code>.metric: str</code> — Distance metric (same as the database).
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
        Shorthand for <code>vault.info().count</code>. Number of live records
        (excludes tombstoned / erased records).
      </p>

      <h3>
        <code>{"Vault.records() -> dict"}</code>
      </h3>
      <p>
        Returns a copy of the internal record store as a plain Python dict
        keyed by record ID. The copy is taken under the read lock, making it
        safe to iterate without holding any lock. Intended for debugging and
        small vaults; avoid on large vaults (&gt; 100 k records).
      </p>

      <h3>
        <code>Vault.pending_removals: int</code>
      </h3>
      <p>
        Number of records that have been erased but not yet physically purged.
        A high value here (relative to <code>count</code>) suggests it is time
        to call <code>vault.vacuum()</code>.
      </p>

      <h3>
        <code>Vault.read_only: bool</code>
      </h3>
      <p>
        <code>True</code> if the vault is in read-only mode. All mutating
        operations raise <code>PermissionError</code> when{" "}
        <code>read_only</code> is <code>True</code>.
      </p>

      <h3>
        <code>{"Vault.set_read_only(value: bool) -> None"}</code>
      </h3>
      <p>
        Dynamically toggle the vault's read-only mode. Useful for building
        ingest/serve pipelines where you want to prevent accidental writes
        during a query phase.
      </p>

      <h3>
        <code>Vault.sealed: bool</code>
      </h3>
      <p>
        When <code>True</code>, the vault accepts no new records (
        <code>place()</code> raises <code>VaultSealed</code>) but searches
        remain available. Sealing is permanent for the lifetime of the
        database file; it is intended for archival vaults where the record set
        is finalised.
      </p>

      <h3>
        <code>{"Vault.rebuild_index() -> None"}</code>
      </h3>
      <p>
        Forces a complete rebuild of the HNSW graph from the current live
        record set. Useful after a large batch erase to restore search quality.
        This operation is blocking and exclusive — no concurrent reads or writes
        are permitted on this vault during the rebuild.
      </p>

      <h3>
        <code>{"Vault.vacuum() -> None"}</code>
      </h3>
      <p>
        Physically removes the storage for erased records within this vault and
        resets <code>pending_removals</code> to 0. Does not rebuild the graph
        index. Call <code>rebuild_index()</code> afterwards if search quality
        has degraded.
      </p>

      {/* ── Result Shape ───────────────────────────────────────── */}
      <h2 id="result-shape">Result Shape</h2>
      <p>
        <code>seek()</code>, <code>seek_text()</code>, and{" "}
        <code>seek_hybrid()</code> all return a list of <code>Result</code>{" "}
        objects (not plain dicts). Each <code>Result</code> has the following
        attributes:
      </p>

      <ul>
        <li>
          <code>id: str</code> — Record identifier.
        </li>
        <li>
          <code>distance: float</code> — Distance from the query vector (lower
          is more similar for cosine and euclidean; higher for dot product).
        </li>
        <li>
          <code>data: dict</code> — Metadata dict stored with the record.
        </li>
        <li>
          <code>document: DocumentAttachment | None</code> — The original text
          document, if one was stored. Has a <code>.text</code> attribute.
        </li>
        <li>
          <code>chunk: ChunkInfo | None</code> — Chunk provenance, if stored.
          Fields: <code>.source</code>, <code>.offset</code>,{" "}
          <code>.page</code>, etc.
        </li>
        <li>
          <code>lineage: EmbeddingLineage | None</code> — Embedding provenance,
          if stored. Fields: <code>.provider</code>, <code>.model</code>,{" "}
          <code>.revision</code>.
        </li>
      </ul>

      <CodeBlock lang="python">{`results = vault.seek(query_vec, top=5)
for r in results:
    print(f"id={r.id}  dist={r.distance:.4f}  data={r.data}")
    if r.document:
        print(f"  text: {r.document.text[:80]}")
    if r.lineage:
        print(f"  embedded by: {r.lineage.provider}/{r.lineage.model}")
`}</CodeBlock>

      <p>
        <strong>scan()</strong> and <strong>fetch()</strong> return plain dicts,
        not <code>Result</code> objects. These dicts contain the same fields
        plus a <code>"vector"</code> key with the raw embedding.
      </p>

      {/* ── Thread Safety ──────────────────────────────────────── */}
      <h2 id="thread-safety">Thread Safety</h2>
      <p>
        Each <code>Vault</code> uses an internal <code>SharedMutex</code>{" "}
        (readers-writer lock):
      </p>
      <ul>
        <li>
          <strong>Read operations</strong> — <code>seek()</code>,{" "}
          <code>seek_text()</code>, <code>seek_hybrid()</code>,{" "}
          <code>scan()</code>, <code>fetch()</code>, <code>info()</code>,{" "}
          <code>count</code> — acquire a <em>shared</em> (read) lock. Multiple
          threads may execute these concurrently without blocking each other.
        </li>
        <li>
          <strong>Write operations</strong> — <code>place()</code>,{" "}
          <code>place_many()</code>, <code>erase()</code>,{" "}
          <code>vacuum()</code>, <code>rebuild_index()</code>, transaction
          commits — acquire an <em>exclusive</em> (write) lock. A write blocks
          until all in-progress reads finish, then subsequent reads block until
          the write completes.
        </li>
        <li>
          <strong>
            <code>set_read_only()</code>
          </strong>{" "}
          — Also takes an exclusive lock.
        </li>
      </ul>
      <p>
        This model is optimised for read-heavy workloads. If your application
        is write-heavy (continuous ingestion), consider batching inserts via{" "}
        <code>place_many()</code> to amortise lock contention.
      </p>

      {/* ── Full Examples ──────────────────────────────────────── */}
      <h2 id="examples">Full Examples</h2>

      <h3>Document ingestion pipeline</h3>
      <CodeBlock lang="python">{`import elips, itertools

db = elips.open(
    "/data/docs.elips",
    dimension=1536,
    metric="cosine",
    embedder=my_embedder,
)
docs_vault = db.vault("docs")

def ingest_chunks(chunks: list[dict], batch_size: int = 500):
    """Embed and ingest text chunks in batches."""
    it = iter(chunks)
    for batch in iter(lambda: list(itertools.islice(it, batch_size)), []):
        records = [
            {
                "text":     c["text"],
                "data":     {"source": c["file"], "page": c["page"]},
                "chunk":    {"source": c["file"], "offset": c["offset"]},
            }
            for c in batch
        ]
        docs_vault.place_many(records)
    print(f"Total: {docs_vault.count} records")
`}</CodeBlock>

      <h3>Semantic search with filter</h3>
      <CodeBlock lang="python">{`from elips import Filter

def search_products(query: str, category: str, max_price: float, n: int = 10):
    vec = my_embedder(query)
    return products.seek(
        vec,
        top=n,
        where=Filter(category=category, price__lte=max_price, in_stock=True),
        threshold=0.5,
    )

hits = search_products("waterproof hiking boots", "footwear", 200.0)
for h in hits:
    print(h.data["sku"], h.distance)
`}</CodeBlock>

      <h3>Periodic maintenance</h3>
      <CodeBlock lang="python">{`def maybe_vacuum(vault, ratio_threshold: float = 0.15):
    """Vacuum if erased records exceed a ratio of total."""
    total = vault.count + vault.pending_removals
    if total > 0 and vault.pending_removals / total > ratio_threshold:
        vault.vacuum()
        # Optionally rebuild for best graph quality
        vault.rebuild_index()
`}</CodeBlock>
    </DocsShell>
  );
}
