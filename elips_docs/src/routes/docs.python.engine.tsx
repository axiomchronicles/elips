import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/engine")({
  head: () => ({
    meta: [
      { title: "Engine — ELIPS Python Reference" },
      {
        name: "description",
        content:
          "Complete reference for the Engine class — the high-level Python wrapper around Database that provides arena management, lifecycle control, and WAL introspection.",
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
        { id: "constructor", label: "Constructor" },
        { id: "connect", label: "connect()" },
        { id: "connect-with-config", label: "connect_with_config()" },
        { id: "properties", label: "Properties" },
        { id: "arena", label: "arena()" },
        { id: "lifecycle", label: "Lifecycle methods" },
        { id: "vault-names", label: "vault_names()" },
        { id: "pending-writes", label: "pending_writes()" },
        { id: "context-manager", label: "Context manager" },
        { id: "thread-safety", label: "Thread safety" },
        { id: "pitfalls", label: "Common mistakes" },
      ]}
    >
      <p className="text-[18px] text-ink">
        <code>Engine</code> is the high-level entry point for the modern ELIPS
        Python API. It wraps a low-level{" "}
        <Link to="/docs/python-sdk">
          <code>Database</code>
        </Link>{" "}
        handle and adds typed arena management, WAL introspection, and an
        idiomatic context-manager protocol. Most applications should open a
        database via <code>elips.connect()</code> rather than constructing{" "}
        <code>Engine</code> directly.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        ELIPS ships two Python surfaces over the same C++ core. The low-level
        surface (<code>open</code> / <code>Database</code> / <code>Vault</code>)
        mirrors the runtime exactly. The modern surface (<code>connect</code> /{" "}
        <code>Engine</code> / <code>Arena</code>) layers typed, text-first
        ergonomics on top. <code>Engine</code> is the bridge: it holds the{" "}
        <code>Database</code> handle, carries an optional default embedder, and
        manufactures <Link to="/docs/python/arena"><code>Arena</code></Link>{" "}
        wrappers on demand via <code>engine.arena(name)</code>.
      </p>
      <p>
        Vaults (arenas) are created lazily — calling{" "}
        <code>engine.arena("documents")</code> is sufficient to bring a vault
        into existence the moment the first record is written to it. No explicit
        schema creation step is required.
      </p>
      <CodeBlock lang="python">{`import elips

# Minimal in-memory database
with elips.connect(":memory:", dimension=128) as engine:
    arena = engine.arena("documents")
    key = arena.write(text="Hello, ELIPS!", meta={"source": "readme"})
    hits = arena.probe_text("Hello", top=3)
    print(hits[0].text)   # Hello, ELIPS!`}</CodeBlock>

      <h2 id="constructor">Constructor</h2>
      <CodeBlock lang="python">{`class Engine:
    def __init__(
        self,
        db: Database,
        *,
        default_embedder: Embedder | None = None,
    ) -> None: ...`}</CodeBlock>
      <p>
        Direct construction is rarely needed. Prefer{" "}
        <code>elips.connect()</code> or{" "}
        <code>elips.connect_with_config()</code> which build the underlying{" "}
        <code>Config</code>, open the database, and return a fully configured{" "}
        <code>Engine</code>.
      </p>
      <ul>
        <li>
          <code>db</code> — An open <code>elips.Database</code> handle obtained
          via <code>elips.open()</code> or <code>elips.open_with_config()</code>
          .
        </li>
        <li>
          <code>default_embedder</code> — Optional Python batch embedder (a
          callable matching the{" "}
          <Link to="/docs/python/models">
            <code>Embedder</code>
          </Link>{" "}
          protocol). Passed down to every arena that does not supply its own
          override. Used when the database has no native text embedder
          configured.
        </li>
      </ul>

      <h2 id="connect">
        <code>connect()</code>
      </h2>
      <CodeBlock lang="python">{`def connect(
    path: str,
    *,
    dimension: int = 0,
    metric: "cosine" | "euclidean" | "dot_product" = "cosine",
    index: "graph" | "exact" = "graph",
    access_mode: "read_write" | "read_only" = "read_write",
    segmented_storage: bool = True,
    metadata_acceleration: bool = True,
    embedder: Embedder | LocalEmbedderConfig | None = None,
    embedder_provider: str = "python",
    embedder_model: str = "callable",
    embedder_revision: str = "",
    use_default_text_embedder: bool = True,
    gpu: GpuConfig | None = None,
) -> Engine`}</CodeBlock>
      <p>
        The canonical way to open an ELIPS database with the modern API. Builds
        a <code>Config</code>, opens the database, attaches the embedder, and
        returns a ready-to-use <code>Engine</code>.
      </p>
      <ul>
        <li>
          <code>path</code> — Filesystem directory path or{" "}
          <code>":memory:"</code> for a transient in-memory database.
        </li>
        <li>
          <code>dimension</code> — Vector dimension for <em>new</em> databases.
          Existing persistent databases restore their dimension from the manifest
          automatically; passing <code>0</code> is safe. In-memory databases
          always need a non-zero dimension.
        </li>
        <li>
          <code>metric</code> — Similarity metric:{" "}
          <code>"cosine"</code> (default), <code>"euclidean"</code>, or{" "}
          <code>"dot_product"</code>.
        </li>
        <li>
          <code>index</code> — Index backend: <code>"graph"</code> (HNSW,
          default) or <code>"exact"</code> (brute-force).
        </li>
        <li>
          <code>access_mode</code> — <code>"read_write"</code> acquires an
          exclusive advisory lock; <code>"read_only"</code> takes a shared lock
          and refuses all mutations.
        </li>
        <li>
          <code>segmented_storage</code> — Whether to use the segmented
          persistence layout (<code>elips.manifest</code> + per-vault segment
          files). Almost always <code>True</code>.
        </li>
        <li>
          <code>metadata_acceleration</code> — Enables the{" "}
          <code>MetadataIndex</code> for equality and set-membership filters.
          Greatly speeds up filtered searches at a small memory cost.
        </li>
        <li>
          <code>embedder</code> — A Python callable or a{" "}
          <code>LocalEmbedderConfig</code>. When a callable is supplied,
          metadata about it is persisted but the callable itself is not.
          Reopening the database without the same callable leaves text-first
          calls raising <code>ValueError</code>.
        </li>
        <li>
          <code>embedder_provider</code> / <code>embedder_model</code> /{" "}
          <code>embedder_revision</code> — Metadata stored alongside a Python
          callable embedder for traceability in{" "}
          <Link to="/docs/python/models">
            <code>EmbeddingLineage</code>
          </Link>
          .
        </li>
        <li>
          <code>use_default_text_embedder</code> — When <code>True</code>
          (default), a new database automatically provisions the built-in local
          text embedder. Set to <code>False</code> when you supply your own{" "}
          <code>embedder</code> or want a vector-only database.
        </li>
        <li>
          <code>gpu</code> — Optional{" "}
          <Link to="/docs/python/gpu">
            <code>GpuConfig</code>
          </Link>{" "}
          for GPU-accelerated index builds. Requires an ELIPS build with GPU
          support.
        </li>
      </ul>
      <CodeBlock lang="python">{`import elips

# Persistent, cosine, HNSW — typical production setup
engine = elips.connect(
    "/var/lib/myapp/vectors",
    dimension=768,
    metric="cosine",
    index="graph",
)

# In-memory — tests and notebooks
engine = elips.connect(":memory:", dimension=128)

# Bring your own embedder (sentence-transformers example)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()

engine = elips.connect(
    "/var/lib/myapp/vectors",
    dimension=384,
    embedder=embed,
    embedder_model="all-MiniLM-L6-v2",
    use_default_text_embedder=False,
)

engine.close()`}</CodeBlock>

      <h2 id="connect-with-config">
        <code>connect_with_config()</code>
      </h2>
      <CodeBlock lang="python">{`def connect_with_config(
    path: str,
    config: Config,
    *,
    embedder: Embedder | LocalEmbedderConfig | None = None,
    embedder_provider: str = "python",
    embedder_model: str = "callable",
    embedder_revision: str = "",
) -> Engine`}</CodeBlock>
      <p>
        Use this when you need fine-grained control over <code>Config</code>{" "}
        options that <code>connect()</code> does not expose directly — for
        example, a <code>LocalEmbedderConfig</code> with a custom model path, or
        GPU tuning parameters that go into the config builder.
      </p>
      <p>
        If <code>config</code> already contains a text embedder,{" "}
        <code>embedder</code> is ignored unless it is a non-local callable, in
        which case it becomes the runtime embedder for Python-side embedding
        fallback.
      </p>
      <CodeBlock lang="python">{`import elips

config = (
    elips.Config()
    .dimension(768)
    .metric("cosine")
    .segmented_storage(True)
    .metadata_acceleration(True)
    .auto_text_embedder(False)   # we bring our own
)

engine = elips.connect_with_config(
    "/var/lib/myapp/vectors",
    config,
    embedder=embed,
    embedder_model="all-MiniLM-L6-v2",
)
engine.close()`}</CodeBlock>

      <h2 id="properties">Properties</h2>
      <h3>
        <code>engine.raw</code> → <code>Database</code>
      </h3>
      <p>
        Returns the underlying low-level <code>Database</code> handle. Useful
        when you need a capability that <code>Engine</code> does not wrap, such
        as <code>db.begin_transaction()</code>, <code>db.query(eql)</code>, or{" "}
        <code>db.gpu_info()</code>.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)

# Drop to low-level for a transaction
with engine.raw.begin_transaction() as txn:
    txn.vault("logs").place([1.0, 0.0], {"msg": "start"})
    txn.vault("logs").place([0.0, 1.0], {"msg": "end"})

engine.close()`}</CodeBlock>

      <h3>
        <code>engine.config</code> → <code>Config</code>
      </h3>
      <p>
        Returns the effective <code>Config</code> as resolved by the runtime —
        including persisted dimension, metric, index type, and embedder
        metadata. Read-only; modifying the returned object has no effect.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=128, metric="euclidean")
cfg = engine.config
print(cfg.dimension_val)          # 128
print(cfg.metric_val)             # euclidean
print(cfg.has_text_embedder)      # True (default embedder attached)
engine.close()`}</CodeBlock>

      <h2 id="arena">
        <code>arena()</code>
      </h2>
      <CodeBlock lang="python">{`engine.arena(
    name: str,
    *,
    embedder: Embedder | None = None,
    text_slot: str = "__elips_text__",
) -> Arena`}</CodeBlock>
      <p>
        Create a typed{" "}
        <Link to="/docs/python/arena">
          <code>Arena</code>
        </Link>{" "}
        wrapper for the named vault. The vault is created lazily — it
        materialises on the first write, not on this call.
      </p>
      <ul>
        <li>
          <code>name</code> — Vault name. Any string is valid; convention is
          lowercase with hyphens (e.g. <code>"documents"</code>,{" "}
          <code>"product-chunks"</code>).
        </li>
        <li>
          <code>embedder</code> — Arena-level embedder override. Takes
          precedence over the engine's <code>default_embedder</code>. Useful
          when different arenas embed in different vector spaces or use different
          models.
        </li>
        <li>
          <code>text_slot</code> — Reserved backward-compat argument. The
          current runtime stores text on{" "}
          <code>DocumentAttachment</code> rather than mirroring it into
          metadata. Leave as default.
        </li>
      </ul>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)

# Two arenas, same database
docs   = engine.arena("documents")
images = engine.arena("image-captions")

docs.write(text="A design document", meta={"kind": "design"})
images.write(vector=[0.5, 0.5], meta={"caption": "hero banner"})

print(engine.vault_names())   # ['documents', 'image-captions']
engine.close()`}</CodeBlock>

      <h2 id="lifecycle">Lifecycle methods</h2>

      <h3>
        <code>engine.checkpoint()</code> → <code>None</code>
      </h3>
      <p>
        Flush the current in-memory database state to durable storage. Writes
        the manifest and segment files (or a snapshot) and truncates the WAL.
        After a checkpoint, <code>engine.pending_writes()</code> returns an
        empty list.
      </p>
      <p>
        <code>close()</code> always checkpoints before releasing locks, so
        manual checkpointing is only needed for long-running write sessions where
        you want to reduce WAL size or guarantee durability mid-session.
      </p>
      <CodeBlock lang="python">{`import tempfile, elips

path = tempfile.mkdtemp()
engine = elips.connect(path, dimension=2)
arena = engine.arena("docs")
for i in range(1000):
    arena.write(vector=[float(i), 0.0])

engine.checkpoint()           # flush now; WAL is truncated
print(engine.pending_writes())   # []
engine.close()`}</CodeBlock>

      <h3>
        <code>engine.compact()</code> → <code>None</code>
      </h3>
      <p>
        Rebuild every vault index from scratch and then checkpoint. Compaction
        produces a higher-quality HNSW graph than incremental inserts, which
        trade quality for throughput. Run after a large bulk load to recover
        optimal recall.
      </p>
      <p>
        Compaction is CPU-intensive and holds the write lock for its full
        duration. Schedule it during low-traffic windows.
      </p>
      <CodeBlock lang="python">{`# After a large bulk import:
keys = arena.ingest(texts=corpus_texts, meta=corpus_meta)
engine.compact()   # rebuild index, then checkpoint`}</CodeBlock>

      <h3>
        <code>engine.vacuum()</code> → <code>None</code>
      </h3>
      <p>
        Reclaim index space held by deleted records across every arena. Deletes
        leave tombstones in the HNSW graph so that live neighbours remain
        reachable; the index widens its beam to bypass them. Tombstones are
        reclaimed automatically once they reach the arena's{" "}
        <code>compaction_ratio</code> (default 0.2), but after a bulk delete it
        is worth reclaiming immediately.
      </p>
      <p>
        Unlike <code>compact()</code>, <code>vacuum()</code> does not rewrite
        the on-disk snapshot and works on in-memory databases.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("documents")
keys = [arena.write(vector=[float(i), 1.0]) for i in range(50)]

# Delete 30 records
arena.discard(keys[:30])
print(arena.pending_removals)   # may be up to 30

engine.vacuum()
print(arena.pending_removals)   # 0
engine.close()`}</CodeBlock>

      <h3>
        <code>engine.close()</code> → <code>None</code>
      </h3>
      <p>
        Checkpoint, release the cross-process advisory lock, and seal every
        arena. After <code>close()</code>, writes to any arena raise{" "}
        <code>elips.StorageError</code> rather than silently failing to persist.
        Calling <code>close()</code> more than once is safe (idempotent).
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("documents")
arena.write(vector=[1.0, 0.0])
engine.close()

# Further writes raise StorageError:
try:
    arena.write(vector=[0.0, 1.0])
except elips.StorageError as exc:
    print("sealed:", exc)`}</CodeBlock>

      <h2 id="vault-names">
        <code>vault_names()</code> → <code>list[str]</code>
      </h2>
      <p>
        Return the names of every vault that currently exists in the database.
        An arena that has been obtained via <code>engine.arena()</code> but
        never written to does not appear here — vaults are created on first
        write.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
print(engine.vault_names())   # []

engine.arena("alpha").write(vector=[1.0, 0.0])
engine.arena("beta").write(vector=[0.0, 1.0])
print(engine.vault_names())   # ['alpha', 'beta']
engine.close()`}</CodeBlock>

      <h2 id="pending-writes">
        <code>pending_writes()</code> → <code>list[WalRecord]</code>
      </h2>
      <p>
        Read the database's write-ahead log without mutating anything. Returns
        every acknowledged record in log order. Transaction markers are resolved
        during replay and never surface here; records inside an unterminated
        transaction are omitted; a corrupt tail is dropped silently.
      </p>
      <p>
        Returns an empty list for <code>":memory:"</code> databases (no WAL
        file) and after <code>checkpoint()</code> (WAL is truncated). Each item
        is a{" "}
        <Link to="/docs/python/models">
          <code>WalRecord</code>
        </Link>
        .
      </p>
      <CodeBlock lang="python">{`import tempfile, elips

path = tempfile.mkdtemp()
engine = elips.connect(path, dimension=2)
arena = engine.arena("docs")
k1 = arena.write(vector=[1.0, 0.0], meta={"rev": 1})
k2 = arena.write(vector=[0.5, 0.5], meta={"rev": 2})

records = engine.pending_writes()
print(len(records))                  # 2
print(records[0].op)                 # insert
print(records[0].arena)              # docs
print(records[0].key == k1)          # True

# After checkpoint, WAL is empty
engine.checkpoint()
print(engine.pending_writes())       # []
engine.close()`}</CodeBlock>
      <p>
        This is the same data that{" "}
        <Link to="/docs/python/wal">
          <code>elips.replay_wal()</code>
        </Link>{" "}
        reads at the file level, wrapped in typed{" "}
        <code>WalRecord</code> objects.
      </p>

      <h2 id="context-manager">Context manager</h2>
      <p>
        <code>Engine</code> implements <code>__enter__</code> and{" "}
        <code>__exit__</code>, making it safe to use with Python's{" "}
        <code>with</code> statement. <code>__exit__</code> always calls{" "}
        <code>close()</code> regardless of whether the block raised an
        exception.
      </p>
      <CodeBlock lang="python">{`import elips

# The context manager is the idiomatic production pattern
with elips.connect("/var/lib/myapp/vectors", dimension=768) as engine:
    arena = engine.arena("documents")
    arena.write_many([
        elips.RecordInput(text="First doc",  meta={"id": 1}),
        elips.RecordInput(text="Second doc", meta={"id": 2}),
    ])
    hits = arena.probe_text("first", top=5)
    for h in hits:
        print(h.key, h.distance, h.text)
# engine.close() is called automatically here`}</CodeBlock>
      <p>
        For long-lived server processes, hold the <code>Engine</code> for the
        lifetime of the process and call <code>close()</code> in a shutdown
        hook, rather than using a context manager.
      </p>

      <h2 id="thread-safety">Thread safety</h2>
      <p>
        <code>Engine</code> itself is a thin Python wrapper and carries no
        thread-local state. The underlying C++ <code>Database</code> serializes
        concurrent writes via an internal mutex. Concurrent reads are safe
        across threads. Concurrent writes from multiple Python threads to the
        same <code>Engine</code> are safe but may contend on the C++ mutex.
      </p>
      <p>
        For multi-process deployments: only one process may hold the{" "}
        <code>read_write</code> lock at a time. Use{" "}
        <code>access_mode="read_only"</code> in reader processes (see{" "}
        <Link to="/docs/python/connect">Connect guide</Link>).
      </p>
      <CodeBlock lang="python">{`import threading, elips

engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("shared")

def worker(i: int) -> None:
    arena.write(vector=[float(i), 0.0], meta={"worker": i})

threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
for t in threads: t.start()
for t in threads: t.join()

print(arena.count())  # 8
engine.close()`}</CodeBlock>

      <h2 id="pitfalls">Common mistakes</h2>
      <ul>
        <li>
          <strong>Writing after close.</strong> Obtaining an{" "}
          <code>Arena</code> and keeping a reference to it after{" "}
          <code>engine.close()</code> will cause the next write to raise{" "}
          <code>StorageError</code>. Always close the engine after all writes
          are done.
        </li>
        <li>
          <strong>
            Forgetting <code>dimension</code> for in-memory databases.
          </strong>{" "}
          <code>elips.connect(":memory:")</code> requires a non-zero{" "}
          <code>dimension</code> every time it is called — there is no manifest
          to restore from.
        </li>
        <li>
          <strong>Multiple read-write openers.</strong> Opening the same
          database directory with two <code>"read_write"</code> processes raises{" "}
          <code>LockConflict</code>. Readers should use{" "}
          <code>access_mode="read_only"</code>.
        </li>
        <li>
          <strong>Calling arena() but never writing.</strong> An arena obtained
          via <code>engine.arena("name")</code> that never receives a write does
          not appear in <code>vault_names()</code>. The vault is created on
          first write.
        </li>
        <li>
          <strong>Reopening without a callable embedder.</strong> If the
          database was opened with a Python callable embedder, reopening it
          without supplying the same callable leaves <code>probe_text</code> and{" "}
          <code>write</code> (text-only) raising <code>ValueError</code>.
        </li>
      </ul>
    </DocsShell>
  );
}
