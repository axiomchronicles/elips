import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/engine")({
  head: () => ({
    meta: [
      { title: "Engine / Client — ELIPS Python Reference" },
      {
        name: "description",
        content:
          "Reference for the ELIPS Engine base class and its Chroma-style Client / PersistentClient: creating and listing collections, EQL, checkpoint/compact/vacuum, and lifecycle.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Engine"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "open", label: "Client / PersistentClient" },
        { id: "create-collection", label: "create_collection()" },
        { id: "get-collection", label: "get_collection()" },
        { id: "get-or-create", label: "get_or_create_collection()" },
        { id: "list-collections", label: "list_collections()" },
        { id: "collection", label: "collection()" },
        { id: "query-eql", label: "query_eql()" },
        { id: "maintenance", label: "Maintenance" },
        { id: "properties", label: "Properties" },
        { id: "context-manager", label: "Context manager" },
        { id: "deprecated", label: "Deprecated shims" },
        { id: "thread-safety", label: "Thread safety" },
      ]}
    >
      <p className="text-[18px] text-ink">
        <code>Engine</code> is the base class of <code>elips.Client</code> and{" "}
        <code>elips.PersistentClient</code>, the Chroma-style entry points to
        ELIPS. It owns the database handle and manufactures{" "}
        <Link to="/docs/python/arena">
          <code>Collection</code>
        </Link>{" "}
        objects. Most applications never name <code>Engine</code> directly — they
        open a <code>Client</code> or <code>PersistentClient</code>.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        The client surface mirrors Chroma: <code>create_collection</code>,{" "}
        <code>get_collection</code>, <code>get_or_create_collection</code>,{" "}
        <code>list_collections</code>, and <code>collection</code>, plus ELIPS
        maintenance (<code>checkpoint</code>, <code>compact</code>,{" "}
        <code>vacuum</code>, <code>pending_writes</code>), raw{" "}
        <code>query_eql</code>, and a context-manager protocol.
      </p>
      <CodeBlock lang="python">{`import elips

# In-memory client (requires a dimension)
with elips.Client(dimension=128) as client:
    docs = client.get_or_create_collection("documents")
    docs.add(ids=["x"], embeddings=[[0.1] * 128], documents=["Hello, ELIPS!"])
    hits = docs.query(query_embeddings=[0.1] * 128, n_results=3)
    print(hits[0].key)`}</CodeBlock>

      <h2 id="open">Client / PersistentClient</h2>
      <CodeBlock lang="python">{`# Ephemeral, in-memory — REQUIRES a dimension (or a dimension-fixing embedder)
elips.Client(
    *,
    dimension: int = 0,
    metric: "cosine" | "euclidean" | "dot_product" = "cosine",
    index: "graph" | "exact" = "graph",
    access_mode: "read_write" | "read_only" = "read_write",
    embedder=None,
    device: str | None = None,
    fallback: bool = True,
    gpu=None,
    **settings,
) -> Client

# On-disk — dimension restored from the manifest when reopening
elips.PersistentClient(
    path: str,
    *,
    dimension: int = 0,
    metric: str = "cosine",
    **settings,
) -> PersistentClient`}</CodeBlock>
      <p>
        <code>Client</code> is a transient in-memory database and always needs a
        non-zero <code>dimension</code> (or an <code>embedder</code> that fixes
        one). <code>PersistentClient</code> is backed by a directory on disk and
        restores <code>dimension</code> / <code>metric</code> from the manifest
        when reopening.
      </p>
      <ul>
        <li>
          <code>metric</code> — <code>"cosine"</code> (default),{" "}
          <code>"euclidean"</code>, or <code>"dot_product"</code>.
        </li>
        <li>
          <code>index</code> — <code>"graph"</code> (HNSW, default) or{" "}
          <code>"exact"</code> (brute-force, no index overhead).
        </li>
        <li>
          <code>embedder</code> — a Python callable or a{" "}
          <code>LocalEmbedderConfig</code> used to embed <code>query_texts</code>{" "}
          / <code>documents</code>.
        </li>
        <li>
          <code>device</code> / <code>fallback</code> / <code>gpu</code> —
          accelerator selection; see the{" "}
          <Link to="/docs/python/gpu">
            <code>GpuConfig</code>
          </Link>{" "}
          reference. There is no <code>gpu=True</code> flag.
        </li>
      </ul>
      <CodeBlock lang="python">{`import elips

# Persistent, cosine, HNSW — typical production setup
client = elips.PersistentClient(
    "/var/lib/myapp/vectors",
    dimension=768,
    metric="cosine",
    index="graph",
)

# In-memory — tests and notebooks
client = elips.Client(dimension=128)

# Metal GPU on Apple hardware, no CPU fallback
client = elips.Client(dimension=384, device="metal", fallback=False)

client.close()`}</CodeBlock>
      <h2 id="create-collection">
        <code>create_collection()</code> → <code>Collection</code>
      </h2>
      <CodeBlock lang="python">{`client.create_collection(
    name: str,
    metadata: dict | None = None,
    configuration: dict | None = None,
    embedder=None,
) -> Collection`}</CodeBlock>
      <p>
        Create a new collection. Raises if a collection with that name already
        exists.
      </p>
      <CodeBlock lang="python">{`docs = client.create_collection("documents", metadata={"owner": "search-team"})`}</CodeBlock>

      <h2 id="get-collection">
        <code>get_collection()</code> → <code>Collection</code>
      </h2>
      <CodeBlock lang="python">{`client.get_collection(name: str, embedder=None) -> Collection`}</CodeBlock>
      <p>
        Return an existing collection. Raises{" "}
        <code>CollectionNotFoundError</code> if it does not exist.
      </p>
      <CodeBlock lang="python">{`docs = client.get_collection("documents")`}</CodeBlock>

      <h2 id="get-or-create">
        <code>get_or_create_collection()</code> → <code>Collection</code>
      </h2>
      <CodeBlock lang="python">{`client.get_or_create_collection(
    name: str,
    metadata: dict | None = None,
    configuration: dict | None = None,
    embedder=None,
) -> Collection`}</CodeBlock>
      <p>
        Return the named collection, creating it lazily on first write if it does
        not yet exist. This is the recommended entry point for most code.
      </p>
      <CodeBlock lang="python">{`docs = client.get_or_create_collection("documents")`}</CodeBlock>
      <h2 id="list-collections">
        <code>list_collections()</code> → <code>list[str]</code>
      </h2>
      <p>
        Return the names of every collection in the database.{" "}
        <code>collection_names()</code> is an alias.
      </p>
      <CodeBlock lang="python">{`print(client.list_collections())   # ['documents', 'images']
print(client.collection_names())   # same`}</CodeBlock>

      <h2 id="collection">
        <code>collection()</code> → <code>Collection</code>
      </h2>
      <p>
        A lazy accessor equivalent to <code>get_or_create_collection(name)</code>
        . The older <code>arena</code> alias is deprecated.
      </p>
      <CodeBlock lang="python">{`docs = client.collection("documents")`}</CodeBlock>
      <p>
        <code>delete_collection(name)</code> and <code>reset()</code> raise{" "}
        <code>NotImplementedError</code> — ELIPS is append-structured and cannot
        drop a collection or wipe the database in place.
      </p>

      <h2 id="query-eql">
        <code>query_eql()</code>
      </h2>
      <CodeBlock lang="python">{`client.query_eql(eql: str, bindings: dict | None = None)`}</CodeBlock>
      <p>
        Run a raw <Link to="/docs/python/database">EQL</Link> statement against
        the database. Bindings use <code>$name</code> placeholders supplied as a
        dict.
      </p>
      <CodeBlock lang="python">{`hits = client.query_eql(
    'search in documents nearest $q k 10 where kind = "design" yield',
    {"q": query_vector},
)`}</CodeBlock>
      <h2 id="maintenance">Maintenance</h2>
      <ul>
        <li>
          <code>checkpoint()</code> — flush pending writes to durable storage and
          truncate the WAL.
        </li>
        <li>
          <code>compact()</code> — rebuild every collection's index and
          checkpoint; produces a higher-quality graph than incremental inserts.
          Run after a large bulk load.
        </li>
        <li>
          <code>vacuum()</code> — reclaim index space held by deleted records
          across all collections, without rewriting the on-disk snapshot.
        </li>
        <li>
          <code>pending_writes()</code> →{" "}
          <Link to="/docs/python/models">
            <code>list[WalRecord]</code>
          </Link>{" "}
          — read the write-ahead log without mutating it. Empty for in-memory
          databases and after <code>checkpoint()</code>.
        </li>
      </ul>
      <CodeBlock lang="python">{`import tempfile, elips

client = elips.PersistentClient(tempfile.mkdtemp(), dimension=2)
docs = client.get_or_create_collection("docs")
for i in range(1000):
    docs.add(embedding=[float(i), 0.0])

records = client.pending_writes()
print(len(records), records[0].op, records[0].arena)

client.checkpoint()                 # flush; WAL truncated
print(client.pending_writes())      # []

docs.rebuild()
client.compact()                    # rebuild indexes + checkpoint
client.close()`}</CodeBlock>
      <h2 id="properties">Properties</h2>
      <h3>
        <code>client.native</code> → <code>Database</code>
      </h3>
      <p>
        The underlying low-level native{" "}
        <Link to="/docs/python/database">
          <code>Database</code>
        </Link>{" "}
        handle. Drop to it for capabilities the client does not wrap —
        transactions, <code>native.query(eql, bindings)</code>, or{" "}
        <code>native.gpu_info()</code>.
      </p>
      <CodeBlock lang="python">{`with client.native.begin_transaction() as txn:
    v = txn.vault("logs")
    v.place([1.0, 0.0], {"msg": "start"})
    v.place([0.0, 1.0], {"msg": "end"})

print(client.native.gpu_info())`}</CodeBlock>
      <h3>
        <code>client.config</code> → <code>Config</code>
      </h3>
      <p>
        The effective <code>Config</code> resolved by the runtime, including
        persisted dimension, metric, index type, and embedder metadata.
      </p>
      <CodeBlock lang="python">{`cfg = client.config
print(cfg.dimension_val, cfg.metric_val, cfg.has_text_embedder)`}</CodeBlock>

      <h2 id="context-manager">Context manager</h2>
      <p>
        <code>Client</code> and <code>PersistentClient</code> implement the
        context-manager protocol; <code>__exit__</code> always calls{" "}
        <code>close()</code>, which checkpoints and releases locks.
      </p>
      <CodeBlock lang="python">{`import elips

with elips.PersistentClient("/var/lib/myapp/vectors", dimension=768) as client:
    docs = client.get_or_create_collection("documents")
    docs.add(
        ids=["1", "2"],
        embeddings=[[0.1] * 768, [0.2] * 768],
        documents=["First doc", "Second doc"],
    )
    hits = docs.query(query_embeddings=[0.1] * 768, n_results=5)
    for h in hits:
        print(h.key, h.distance, h.text)
# client.close() runs automatically`}</CodeBlock>
      <p>
        For long-lived servers, hold the client for the process lifetime and call{" "}
        <code>close()</code> from a shutdown hook instead.
      </p>
      <h2 id="deprecated">Deprecated shims</h2>
      <p>
        The pre-Chroma entry points still work but emit{" "}
        <code>DeprecationWarning</code>; do not use them in new code.
      </p>
      <ul>
        <li>
          <code>elips.connect(path, ...)</code> and{" "}
          <code>elips.connect_with_config(path, config)</code> — superseded by{" "}
          <code>elips.Client</code> / <code>elips.PersistentClient</code>.
        </li>
        <li>
          <code>elips.Engine</code>, <code>elips.open</code> — the old
          low-level entry points.
        </li>
        <li>
          <code>client.arena(name)</code> — alias for{" "}
          <code>collection(name)</code>.
        </li>
      </ul>
      <p>
        There is no <code>index_type=</code>, no <code>durability=</code>, and no{" "}
        <code>gpu=True</code> keyword on the client constructor.
      </p>

      <h2 id="thread-safety">Thread safety</h2>
      <p>
        The client is a thin wrapper with no thread-local state. The underlying
        native{" "}
        <Link to="/docs/python/database">
          <code>Database</code>
        </Link>{" "}
        serializes concurrent writes via an internal mutex and allows concurrent
        reads. For multi-process use, only one process may hold the{" "}
        <code>read_write</code> lock; open reader processes with{" "}
        <code>access_mode="read_only"</code>.
      </p>
      <CodeBlock lang="python">{`import threading, elips

client = elips.Client(dimension=2)
docs = client.get_or_create_collection("shared")

def worker(i: int) -> None:
    docs.add(embedding=[float(i), 0.0], metadata={"worker": i})

threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
for t in threads: t.start()
for t in threads: t.join()

print(docs.count())   # 8
client.close()`}</CodeBlock>
    </DocsShell>
  );
}
