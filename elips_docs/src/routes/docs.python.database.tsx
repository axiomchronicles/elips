import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/database")({
  head: () => ({
    meta: [
      { title: "Database — ELIPS Python API Docs" },
      {
        name: "description",
        content:
          "Complete reference for the ELIPS Python Database class: opening, configuring, querying, and managing an embedded vector database instance.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="PYTHON API · LOW-LEVEL"
      title="Database"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "opening", label: "Opening a Database" },
        { id: "open-with-config", label: "open_with_config()" },
        { id: "vault", label: "vault()" },
        { id: "list-vaults", label: "list_vaults()" },
        { id: "transactions", label: "begin_transaction()" },
        { id: "query", label: "query() — EQL" },
        { id: "maintenance", label: "Maintenance Operations" },
        { id: "properties", label: "Properties" },
        { id: "thread-safety", label: "Thread Safety" },
        { id: "lifecycle", label: "Lifecycle & Destruction" },
        { id: "examples", label: "Full Examples" },
      ]}
    >
      {/* ── Overview ───────────────────────────────────────────── */}
      <h2 id="overview">Overview</h2>
      <p>
        <code>Database</code> is the top-level handle to an ELIPS instance. In
        the C++ layer it is <code>ElipsInstance</code>; the Python binding
        exposes it under the name <code>Database</code>. Every operation —
        inserting vectors, searching, issuing EQL queries, managing
        transactions — flows through this object.
      </p>
      <p>
        A <code>Database</code> is opened once and kept alive for the lifetime
        of your process (or for as long as you need it). It is safe to share
        across threads; an internal instance mutex guards the vault registry,
        while each individual <code>Vault</code> carries its own
        reader-writer lock that allows concurrent searches without
        serialisation.
      </p>

      <CodeBlock lang="python">{`import elips

# Minimal — in-memory ephemeral database
db = elips.open(":memory:", dimension=128, metric="cosine")

# Persistent on disk
db = elips.open(
    "/var/lib/myapp/vectors.elips",
    dimension=768,
    metric="cosine",
    durability="standard",
)
`}</CodeBlock>

      {/* ── Opening ────────────────────────────────────────────── */}
      <h2 id="opening">Opening a Database — <code>elips.open()</code></h2>
      <p>
        The module-level <code>elips.open()</code> function is the primary
        entry point. It constructs a <code>Config</code> from keyword
        arguments, validates it against any existing on-disk metadata, and
        returns a fully initialised <code>Database</code>.
      </p>

      <h3>Signature</h3>
      <CodeBlock lang="python">{`elips.open(
    path: str,
    *,
    dimension: int                           = 0,
    metric: str                              = "cosine",
    durability: str                          = "standard",
    access_mode: str                         = "read_write",
    index: str                               = "graph",
    segmented_storage: bool                  = False,
    metadata_acceleration: bool              = True,
    embedder: callable | None                = None,
    **kwargs,
) -> Database
`}</CodeBlock>

      <h3>Parameters</h3>
      <ul>
        <li>
          <strong>
            <code>path</code>
          </strong>{" "}
          — Filesystem path to the database directory. Pass{" "}
          <code>":memory:"</code> for a fully in-memory, non-persistent
          database. The directory is created if it does not exist (for
          read-write mode).
        </li>
        <li>
          <strong>
            <code>dimension</code>
          </strong>{" "}
          — Number of dimensions for every vector stored in this database.
          Required when creating a new database. Ignored (but validated) when
          reopening — ELIPS locks the dimension to the value stored in the
          persisted metadata; passing a mismatching value raises{" "}
          <code>ConfigError</code>.
        </li>
        <li>
          <strong>
            <code>metric</code>
          </strong>{" "}
          — Distance metric. One of <code>"cosine"</code>,{" "}
          <code>"euclidean"</code>, or <code>"dot_product"</code>. Like{" "}
          <code>dimension</code>, this is a persisted identity field and cannot
          be changed after the first open.
        </li>
        <li>
          <strong>
            <code>durability</code>
          </strong>{" "}
          — WAL flush strategy. See <a href="/docs/python/config">Config</a>{" "}
          for the full breakdown of each level. Defaults to{" "}
          <code>"standard"</code>.
        </li>
        <li>
          <strong>
            <code>access_mode</code>
          </strong>{" "}
          — <code>"read_write"</code> (default) or <code>"read_only"</code>.
          Read-only mode acquires no file locks beyond a shared reader lock and
          disables all mutation APIs.
        </li>
        <li>
          <strong>
            <code>index</code>
          </strong>{" "}
          — Index strategy: <code>"graph"</code> (HNSW, default) or{" "}
          <code>"exact"</code> (brute-force, no index overhead, scales poorly
          beyond ~100 k vectors).
        </li>
        <li>
          <strong>
            <code>segmented_storage</code>
          </strong>{" "}
          — When <code>True</code>, vector data is stored across multiple
          memory-mapped segments, enabling databases larger than the available
          virtual address space on 32-bit platforms (rare). Defaults to{" "}
          <code>False</code>.
        </li>
        <li>
          <strong>
            <code>metadata_acceleration</code>
          </strong>{" "}
          — Maintains a secondary in-memory hash map of record metadata for
          O(1) filter evaluation during search. Disable only if memory is
          extremely constrained. Defaults to <code>True</code>.
        </li>
        <li>
          <strong>
            <code>embedder</code>
          </strong>{" "}
          — A callable <code>(text: str) {"->"} list[float]</code> that ELIPS
          calls automatically when you use text-based APIs (
          <code>seek_text</code>, <code>place_document</code>). Equivalent to
          calling <code>Config.text_embedder(fn, ...)</code>.
        </li>
      </ul>

      <h3>Return value</h3>
      <p>
        A <code>Database</code> instance. The connection to the storage engine
        is live immediately; no further initialisation call is needed.
      </p>

      <h3>Exceptions</h3>
      <ul>
        <li>
          <code>ConfigError</code> — dimension/metric/index conflict with
          existing database, or an invalid parameter value was supplied.
        </li>
        <li>
          <code>IOError</code> / <code>OSError</code> — path is not accessible
          or the database files are corrupted.
        </li>
        <li>
          <code>LockConflict</code> — another process holds an exclusive lock
          on the database directory.
        </li>
      </ul>

      <CodeBlock lang="python">{`import elips

# New database — dimension is set for the first time
db = elips.open(
    "/data/products.elips",
    dimension=1536,
    metric="cosine",
    durability="paranoid",
    index="graph",
    metadata_acceleration=True,
)

# Reopen the same database — dimension/metric are read from disk
db2 = elips.open("/data/products.elips")

# Wrong dimension → ConfigError
try:
    bad = elips.open("/data/products.elips", dimension=512)
except elips.ConfigError as e:
    print(e)  # Dimension mismatch: expected 1536, got 512
`}</CodeBlock>

      {/* ── open_with_config ───────────────────────────────────── */}
      <h2 id="open-with-config">
        <code>elips.open_with_config(path, config)</code>
      </h2>
      <p>
        A lower-level alternative to <code>elips.open()</code> that accepts a
        pre-built <code>Config</code> object. Useful when you construct{" "}
        <code>Config</code> programmatically — for example, from a TOML file or
        environment variables — and want to avoid keyword argument forwarding.
      </p>

      <CodeBlock lang="python">{`import elips
from elips import Config, GraphParams

cfg = (
    Config()
    .dimension(768)
    .metric("cosine")
    .index("graph")
    .graph_params(GraphParams(max_connections=32, ef_construction=400, ef_search=100))
    .durability("standard")
    .metadata_acceleration(True)
)

db = elips.open_with_config("/data/embeddings.elips", cfg)
`}</CodeBlock>

      <p>
        The function signature is:
      </p>
      <CodeBlock lang="python">{`elips.open_with_config(path: str, config: Config) -> Database
`}</CodeBlock>

      {/* ── vault() ────────────────────────────────────────────── */}
      <h2 id="vault">
        <code>Database.vault(name)</code>
      </h2>
      <p>
        Returns a <code>Vault</code> — a named namespace within the database
        that stores its own set of vectors and associated records. Vaults are
        created lazily: the first call to <code>vault("name")</code> for a given
        name creates and registers the vault; subsequent calls return the same
        object.
      </p>

      <CodeBlock lang="python">{`elips.Database.vault(name: str) -> elips.Vault
`}</CodeBlock>

      <h3>Parameters</h3>
      <ul>
        <li>
          <strong>
            <code>name</code>
          </strong>{" "}
          — A non-empty string. Vault names are case-sensitive. The name is
          persisted and must remain stable across restarts.
        </li>
      </ul>

      <h3>Notes</h3>
      <ul>
        <li>
          The vault registry is protected by the instance mutex, so concurrent
          calls to <code>vault()</code> from multiple threads are safe — the
          first caller creates the vault, subsequent callers receive the cached
          reference.
        </li>
        <li>
          All vaults within a database share the same dimension and metric
          (they are database-level identity fields).
        </li>
        <li>
          A vault with a given name that already exists on disk is loaded; a new
          name results in an empty vault being created in memory and flushed on
          first write.
        </li>
      </ul>

      <CodeBlock lang="python">{`db = elips.open("/data/shop.elips", dimension=384, metric="cosine")

products = db.vault("products")
reviews  = db.vault("reviews")
# Calling vault() again returns the same object
assert db.vault("products") is products
`}</CodeBlock>

      {/* ── list_vaults() ──────────────────────────────────────── */}
      <h2 id="list-vaults">
        <code>Database.list_vaults()</code>
      </h2>
      <p>
        Returns the names of all vaults currently registered in the database,
        including those that exist on disk but have not yet been accessed in
        this process session.
      </p>

      <CodeBlock lang="python">{`Database.list_vaults() -> list[str]
`}</CodeBlock>

      <CodeBlock lang="python">{`names = db.list_vaults()
print(names)  # ['products', 'reviews', 'sessions']
`}</CodeBlock>

      {/* ── begin_transaction() ────────────────────────────────── */}
      <h2 id="transactions">
        <code>Database.begin_transaction()</code>
      </h2>
      <p>
        Returns a <code>Transaction</code> object that groups multiple
        mutations across one or more vaults into a single atomic operation.
        Prefer using it as a context manager so that rollback is automatic on
        error.
      </p>

      <CodeBlock lang="python">{`Database.begin_transaction() -> Transaction
`}</CodeBlock>

      <p>
        See the dedicated{" "}
        <a href="/docs/python/transaction">Transaction reference</a> for full
        details on commit, rollback, WAL framing, and crash-safety guarantees.
      </p>

      <CodeBlock lang="python">{`with db.begin_transaction() as txn:
    v = txn.vault("products")
    v.place([0.1] * 768, data={"sku": "ABC-001"})
    v.place([0.2] * 768, data={"sku": "ABC-002"})
# Committed atomically — both records appear or neither does
`}</CodeBlock>

      {/* ── query() ────────────────────────────────────────────── */}
      <h2 id="query">
        <code>Database.query(eql, bindings={})</code>
      </h2>
      <p>
        Executes an <strong>ELIPS Query Language (EQL)</strong> statement
        against the database and returns the results as a list of{" "}
        <code>Result</code> objects.
      </p>

      <CodeBlock lang="python">{`Database.query(eql: str, bindings: dict = {}) -> list[Result]
`}</CodeBlock>

      <h3>Parameters</h3>
      <ul>
        <li>
          <strong>
            <code>eql</code>
          </strong>{" "}
          — A valid EQL statement string. EQL supports{" "}
          <code>SEEK</code>, <code>SCAN</code>, <code>PLACE</code>,{" "}
          <code>ERASE</code>, and <code>FETCH</code> operations with a
          SQL-inspired syntax.
        </li>
        <li>
          <strong>
            <code>bindings</code>
          </strong>{" "}
          — A mapping from placeholder names to Python values. Use{" "}
          <code>$name</code> placeholders in the EQL string and supply their
          values here to avoid injection vulnerabilities.
        </li>
      </ul>

      <CodeBlock lang="python">{`# Simple seek via EQL
results = db.query(
    "SEEK $vec TOP 10 IN products",
    bindings={"vec": embedding_vector},
)
for r in results:
    print(r.id, r.distance, r.data)

# EQL with a metadata filter
results = db.query(
    "SEEK $vec TOP 5 IN products WHERE category = $cat AND price < $max_price",
    bindings={
        "vec": embedding_vector,
        "cat": "electronics",
        "max_price": 999.0,
    },
)

# SCAN
records = db.query("SCAN products LIMIT 100")
`}</CodeBlock>

      {/* ── Maintenance ────────────────────────────────────────── */}
      <h2 id="maintenance">Maintenance Operations</h2>

      <h3>
        <code>Database.checkpoint()</code>
      </h3>
      <p>
        Flushes all pending WAL entries to the main data files, then truncates
        the WAL. This reduces WAL file size and shortens recovery time on the
        next open. Calling <code>checkpoint()</code> does not compact or
        rebuild the graph index.
      </p>
      <CodeBlock lang="python">{`Database.checkpoint() -> None
`}</CodeBlock>

      <h3>
        <code>Database.compact()</code>
      </h3>
      <p>
        Runs a full compaction pass: checkpoints the WAL, rewrites the data
        segment to remove dead space from erased records, and rebuilds the HNSW
        graph index for each vault whose tombstone ratio exceeds its configured{" "}
        <code>compaction_ratio</code>. This is an expensive, blocking operation
        — schedule it during off-peak periods.
      </p>
      <CodeBlock lang="python">{`Database.compact() -> None
`}</CodeBlock>

      <h3>
        <code>Database.vacuum()</code>
      </h3>
      <p>
        Reclaims disk space by physically removing the storage files for any
        vaults that have been deleted, and truncating free-list space in the
        main data file. Unlike <code>compact()</code>, it does not rebuild
        graph indexes. Typically run after a large number of erases.
      </p>
      <CodeBlock lang="python">{`Database.vacuum() -> None
`}</CodeBlock>

      <h3>
        <code>Database.close()</code>
      </h3>
      <p>
        Explicitly closes the database: flushes all pending writes, writes a
        checkpoint (for persistent databases), releases all file locks, and
        invalidates all <code>Vault</code> handles obtained from this instance.
        After calling <code>close()</code>, any further method call on the
        <code>Database</code> or its <code>Vault</code> objects raises{" "}
        <code>RuntimeError</code>.
      </p>
      <p>
        You do not need to call <code>close()</code> explicitly if you use the
        database as a context manager, or if you allow normal Python garbage
        collection to destroy the object (the destructor checkpoints and
        closes).
      </p>
      <CodeBlock lang="python">{`db.close()

# Or, using Database as a context manager (preferred for scripts):
with elips.open("/data/vectors.elips", dimension=128, metric="cosine") as db:
    vault = db.vault("default")
    vault.place([0.5] * 128, data={"label": "test"})
# Checkpoint and close happen automatically here
`}</CodeBlock>

      {/* ── Properties ─────────────────────────────────────────── */}
      <h2 id="properties">Properties</h2>

      <h3>
        <code>Database.path: str</code>
      </h3>
      <p>
        The filesystem path that was passed to <code>open()</code>. Returns{" "}
        <code>":memory:"</code> for ephemeral databases.
      </p>

      <h3>
        <code>Database.config: Config</code>
      </h3>
      <p>
        The fully-resolved <code>Config</code> in effect for this database.
        Includes values read back from persisted metadata (so{" "}
        <code>db.config.dimension_val</code> is always accurate even if you did
        not pass <code>dimension=</code> at open time).
      </p>

      <CodeBlock lang="python">{`print(db.config.dimension_val)  # e.g. 768
print(db.config.metric_val)     # 'cosine'
print(db.config.index_val)      # 'graph'
`}</CodeBlock>

      <h3>
        <code>Database.is_persistent: bool</code>
      </h3>
      <p>
        <code>True</code> if the database is backed by disk storage (i.e.,{" "}
        <code>path != ":memory:"</code> and durability is not{" "}
        <code>"ephemeral"</code>). Persistent databases are checkpointed on
        destruction.
      </p>

      <h3>
        <code>Database.wal</code>
      </h3>
      <p>
        Low-level access to the Write-Ahead Log handle. Exposes WAL position
        counters and flush statistics. Intended for monitoring and debugging,
        not normal application code. The exact API is an implementation detail
        and subject to change between minor versions.
      </p>

      {/* ── Thread Safety ──────────────────────────────────────── */}
      <h2 id="thread-safety">Thread Safety</h2>
      <p>ELIPS has a layered concurrency model:</p>
      <ul>
        <li>
          <strong>Instance mutex</strong> — A single mutex guards the vault
          registry. Calls to <code>vault()</code> and{" "}
          <code>list_vaults()</code> lock this mutex briefly. It is{" "}
          <em>not</em> held during search or insert.
        </li>
        <li>
          <strong>Per-vault SharedMutex</strong> — Each <code>Vault</code>{" "}
          holds a reader-writer lock. Multiple threads can execute{" "}
          <code>seek()</code> simultaneously on the same vault without
          serialisation. Write operations (<code>place()</code>,{" "}
          <code>erase()</code>, committed transactions) acquire an exclusive
          write lock.
        </li>
        <li>
          <strong>
            <code>Database</code> itself is thread-safe
          </strong>{" "}
          — you may share one <code>Database</code> object across threads
          freely.
        </li>
        <li>
          <strong>
            <code>Transaction</code> is NOT thread-safe
          </strong>{" "}
          — do not share a <code>Transaction</code> between threads. Use one
          transaction per thread.
        </li>
      </ul>

      <CodeBlock lang="python">{`import threading, elips

db = elips.open("/data/vectors.elips", dimension=384, metric="cosine")
vault = db.vault("items")

def search_worker(query_vec):
    # Safe — concurrent seeks do not serialize
    results = vault.seek(query_vec, top=10)
    return results

threads = [threading.Thread(target=search_worker, args=([0.1]*384,)) for _ in range(8)]
for t in threads: t.start()
for t in threads: t.join()
`}</CodeBlock>

      {/* ── Lifecycle ──────────────────────────────────────────── */}
      <h2 id="lifecycle">Lifecycle & Destruction</h2>
      <p>
        When a <code>Database</code> object is garbage-collected or its{" "}
        <code>__del__</code> runs, ELIPS performs the following sequence for
        persistent databases:
      </p>
      <ol>
        <li>Acquire the instance mutex.</li>
        <li>
          Flush all dirty in-memory pages and pending WAL records to disk.
        </li>
        <li>Write a WAL checkpoint record.</li>
        <li>Release all file locks.</li>
        <li>Close memory-mapped file handles.</li>
      </ol>
      <p>
        For ephemeral (<code>":memory:"</code> or{" "}
        <code>durability="ephemeral"</code>) databases, the destructor simply
        discards all in-memory data — no disk I/O occurs.
      </p>
      <p>
        <strong>Best practice:</strong> in long-running server processes,
        keep a single <code>Database</code> alive for the process lifetime.
        Avoid repeated open/close cycles — each open replays the WAL from disk,
        which adds latency proportional to WAL size.
      </p>

      {/* ── Full Examples ──────────────────────────────────────── */}
      <h2 id="examples">Full Examples</h2>

      <h3>Server application pattern</h3>
      <CodeBlock lang="python">{`import elips
import numpy as np

# Module-level singleton — opened once at startup
db = elips.open(
    "/var/lib/myapp/vectors.elips",
    dimension=1536,
    metric="cosine",
    durability="standard",
    metadata_acceleration=True,
)

products = db.vault("products")
sessions = db.vault("sessions")

def index_product(sku: str, embedding: list[float], meta: dict) -> str:
    return products.place(embedding, data={"sku": sku, **meta})

def find_similar(embedding: list[float], n: int = 10):
    return products.seek(embedding, top=n)

# Periodic maintenance — e.g., called from a cron job
def nightly_maintenance():
    db.compact()
`}</CodeBlock>

      <h3>Ephemeral / test database</h3>
      <CodeBlock lang="python">{`import elips

def test_vector_search():
    db = elips.open(":memory:", dimension=4, metric="cosine")
    v = db.vault("test")
    v.place([1.0, 0.0, 0.0, 0.0], data={"label": "x-axis"})
    v.place([0.0, 1.0, 0.0, 0.0], data={"label": "y-axis"})
    results = v.seek([1.0, 0.1, 0.0, 0.0], top=1)
    assert results[0].data["label"] == "x-axis"
    db.close()
`}</CodeBlock>

      <h3>Read-only replica</h3>
      <CodeBlock lang="python">{`# A secondary process can open the same path read-only
reader = elips.open(
    "/var/lib/myapp/vectors.elips",
    access_mode="read_only",
)
results = reader.vault("products").seek(query_vec, top=5)
`}</CodeBlock>
    </DocsShell>
  );
}
