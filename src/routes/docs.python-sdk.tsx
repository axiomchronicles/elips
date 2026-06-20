import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python-sdk")({
  head: () => ({
    meta: [
      { title: "Python SDK — ELIPS Docs" },
      { name: "description", content: "The complete ELIPS Python surface — low-level bindings, the modern Engine/Arena wrapper, embedders, EQL, transactions, and read-only serving." },
      { property: "og:title", content: "Python SDK — ELIPS" },
      { property: "og:description", content: "Database, Vault, Config, Filter, EQL, embedders, and the modern Engine/Arena wrapper." },
      { property: "og:url", content: "/docs/python-sdk" },
    ],
    links: [{ rel: "canonical", href: "/docs/python-sdk" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell eyebrow="Reference · Python" title="Python SDK" toc={[
      { id: "install", label: "Install & build" },
      { id: "two-layers", label: "Two surfaces" },
      { id: "low-level", label: "Low-level API" },
      { id: "modern", label: "Modern wrapper" },
      { id: "config", label: "Config" },
      { id: "ingestion", label: "Ingestion" },
      { id: "query", label: "Query & planner" },
      { id: "eql", label: "EQL from Python" },
      { id: "txn", label: "Transactions" },
      { id: "embedders", label: "Embedders" },
      { id: "lifecycle", label: "Persistence & lifecycle" },
      { id: "errors", label: "Errors" },
      { id: "typing", label: "Typing" },
      { id: "pitfalls", label: "Pitfalls" },
    ]}>
      <p className="text-[18px] text-ink">
        ELIPS ships two Python surfaces over the same C++ core. The
        low-level bindings (<code>open</code> / <code>Database</code> /{" "}
        <code>Vault</code> / <code>Config</code>) mirror the runtime
        one-to-one. The modern wrapper (<code>connect</code> /{" "}
        <code>Engine</code> / <code>Arena</code>) adds typed text-first
        ergonomics on top. Both speak to the same vaults, the same WAL,
        and the same planner.
      </p>

      <h2 id="install">Install &amp; build</h2>
      <p>
        The bindings are built from the repository — there is no PyPI
        wheel yet. Build the extension and put the package on{" "}
        <code>PYTHONPATH</code>:
      </p>
      <CodeBlock lang="bash">
{`cmake -S . -B build -G Ninja -DELIPS_BUILD_PYTHON=ON
cmake --build build --target elips_pymodule
export PYTHONPATH=$PWD/bindings/python`}
      </CodeBlock>

      <h2 id="two-layers">Two surfaces, one core</h2>
      <ul>
        <li><strong>Low-level</strong>: <code>open()</code>, <code>open_with_config()</code>, <code>Database</code>, <code>Vault</code>, <code>Config</code>. Pick this when you need exact parity with the C++ runtime — full control over <code>Durability</code>, <code>AccessMode</code>, GPU settings, and the embedder.</li>
        <li><strong>Modern</strong>: <code>connect()</code>, <code>Engine</code>, <code>Arena</code>, <code>RecordInput</code>, <code>Row</code>, <code>Hit</code>. Pick this for typed, text-first ingestion and retrieval.</li>
      </ul>

      <h2 id="low-level">Low-level API</h2>
      <CodeBlock lang="python">
{`import elips

db = elips.open("/tmp/elips-sdk", dimension=128, metric="cosine")
docs = db.vault("documents")
docs.place_document("alpha design note", {"kind": "design"})
docs.place_document("beta runbook", {"kind": "ops"})

hits = docs.seek_text("alpha", top=2)
print(hits[0].document.text, db.config.text_embedder_info.model)

plan = docs.explain_seek(
    [1.0, 0.0],
    top=1,
    where=elips.Filter().field("kind").equals("design"),
    has_text_component=True,
)
print(plan.strategy, plan.metadata_accelerated)`}
      </CodeBlock>

      <h3><code>elips.open()</code></h3>
      <CodeBlock lang="python">
{`def open(
    path: str,
    dimension: int = 0,
    metric: str = "cosine",
    index: str = "graph",
    access_mode: str = "read_write",
    embedder = None,
    use_default_text_embedder: bool = True,
) -> Database`}
      </CodeBlock>
      <ul>
        <li><code>path</code> — filesystem directory or <code>":memory:"</code>.</li>
        <li><code>dimension</code> — required for new databases and every in-memory open. Existing databases reuse the persisted identity.</li>
        <li><code>metric</code> — <code>"cosine"</code>, <code>"euclidean"</code>, or <code>"dot_product"</code>.</li>
        <li><code>index</code> — <code>"graph"</code> (HNSW) or <code>"exact"</code>.</li>
        <li><code>access_mode</code> — <code>"read_write"</code> or <code>"read_only"</code>; read-only requires an existing database.</li>
        <li><code>embedder</code> — optional Python callable or <code>LocalEmbedderConfig</code>.</li>
        <li><code>use_default_text_embedder</code> — attach the built-in local embedder automatically when no explicit embedder is supplied.</li>
      </ul>

      <h3><code>Database</code></h3>
      <ul>
        <li><code>vault(name)</code> — return (lazily creating) a <code>Vault</code>.</li>
        <li><code>list_vaults()</code> — current vault names.</li>
        <li><code>begin_transaction()</code> — atomic batched writes.</li>
        <li><code>query(eql, bindings={'{}'})</code> — execute one EQL statement.</li>
        <li><code>checkpoint()</code> — flush manifest/segments or snapshot and truncate the WAL.</li>
        <li><code>compact()</code> — rebuild every vault index and checkpoint.</li>
        <li><code>close()</code> — graceful shutdown; idempotent.</li>
        <li><code>abandon()</code> — testing hook that suppresses checkpoint so the next open must recover from the WAL.</li>
        <li><code>config</code> — effective <code>Config</code> including persisted dimension/metric/index and resolved embedder metadata.</li>
        <li><code>gpu_info()</code> / <code>gpu_stats()</code> — available only in GPU builds.</li>
      </ul>

      <h3><code>Vault</code></h3>
      <CodeBlock lang="python">
{`rid = docs.place(
    [1.0, 0.0],
    {"kind": "design"},
    document=elips.DocumentAttachment(text="alpha design note"),
    chunk=chunk,
    lineage=lineage,
)

# Text-first — requires a configured text embedder
rid = docs.place_document("alpha design note", {"kind": "design"})

# Mixed batch
docs.place_many([
    {"vector": [1.0, 0.0], "data": {"kind": "vector-only"}},
    {"text": "alpha design note", "data": {"kind": "text-first"}},
])`}
      </CodeBlock>
      <p>
        Query surfaces — <code>seek</code>, <code>seek_text</code>,{" "}
        <code>seek_hybrid</code> — all accept <code>top</code>,{" "}
        <code>where</code>, and <code>threshold</code>. Hybrid takes an
        extra <code>lexical_weight</code> (default <code>0.25</code>).
        Every hit returns <code>id</code>, <code>distance</code>,{" "}
        <code>data</code>, <code>document</code>, <code>chunk</code>, and{" "}
        <code>lineage</code> from the authoritative record store.
      </p>

      <h2 id="modern">Modern wrapper</h2>
      <CodeBlock lang="python">
{`engine = elips.connect(
    "/tmp/elips-modern",
    dimension=128,
    metric="cosine",
)
arena = engine.arena("documents")

keys = arena.write_many([
    elips.RecordInput(text="alpha design note", meta={"kind": "design"}),
    {"text": "beta runbook", "meta": {"kind": "ops"}},
])

rows = arena.pull(keys, include_vectors=True)
hits = arena.probe_text("alpha", top=2)
hybrid = arena.probe_hybrid([0.0, 1.0], "alpha", top=2)`}
      </CodeBlock>
      <p>
        <code>Arena</code> prefers the native core text APIs when the
        database config has a resolved text embedder. If the wrapper is
        given a Python callable instead, it falls back to Python-side
        embedding plus <code>seek_hybrid</code>. If neither exists,
        text-first calls raise <code>ConfigError</code> /{" "}
        <code>ValueError</code> — ELIPS never silently degrades to
        lexical-only behaviour.
      </p>

      <h2 id="config">Config</h2>
      <CodeBlock lang="python">
{`config = (
    elips.Config()
    .dimension(2)
    .metric("cosine")
    .segmented_storage(True)
    .metadata_acceleration(True)
    .auto_text_embedder(True)
)
db = elips.open_with_config("/tmp/elips-quickstart", config)`}
      </CodeBlock>
      <ul>
        <li><code>segmented_storage(True)</code> — default. Writes <code>elips.manifest</code> + per-vault segment files.</li>
        <li><code>metadata_acceleration(True)</code> — equality and set-membership filters narrow candidates through <code>MetadataIndex</code>.</li>
        <li><code>auto_text_embedder(True)</code> — provisions the built-in local embedder for new databases.</li>
        <li><code>local_text_embedder(...)</code> — pin a rehydratable local embedder that ELIPS restores automatically on reopen.</li>
        <li><code>text_embedder(callable, ...)</code> — attach a Python callable embedder. ELIPS persists metadata only; reopening without the same callable makes text-first APIs raise <code>ConfigError</code>.</li>
      </ul>

      <h2 id="ingestion">Ingestion patterns</h2>
      <CodeBlock lang="python">
{`# 1. Vector with attached document
attachment = elips.DocumentAttachment(text="gamma appendix", mime_type="text/plain")
docs.place([1.0, 0.0], {"kind": "appendix"}, document=attachment)

# 2. Text-first
docs.place_document("alpha design note", {"kind": "design"})

# 3. Explicit chunk coordinates
chunk = elips.ChunkInfo()
chunk.document_key = "doc-alpha"
chunk.ordinal = 0
chunk.char_start = 0
chunk.char_end = 17
docs.place_document("alpha design note", {"kind": "design"}, chunk=chunk)`}
      </CodeBlock>

      <h2 id="query">Query &amp; planner</h2>
      <CodeBlock lang="python">
{`# Vector
hits = docs.seek([1.0, 0.0], top=2)

# Text-first
hits = docs.seek_text("alpha", top=2)

# Hybrid (vector + lexical overlap from attached documents)
hits = docs.seek_hybrid([0.0, 1.0], "alpha", top=2, lexical_weight=0.35)

# Inspect the plan
where = elips.Filter().field("kind").equals("design")
plan = docs.explain_seek([1.0, 0.0], top=1, where=where, has_text_component=True)
print(plan.strategy.name, plan.metadata_accelerated, plan.candidate_count)`}
      </CodeBlock>
      <p>
        The planner always emits one of{" "}
        <code>ann_index</code>, <code>exact_candidates</code>,{" "}
        <code>full_scan</code>, <code>text_probe</code>,{" "}
        <code>hybrid_fusion</code> — exposed identically here and in EQL.
      </p>

      <h2 id="eql">EQL from Python</h2>
      <CodeBlock lang="python">
{`rows = db.query(
    "seek in documents nearest $q top 5 where kind = \\"design\\" yield",
    bindings={"q": [1.0, 0.0]},
)`}
      </CodeBlock>
      <p>
        Text-first retrieval is not exposed through EQL — use{" "}
        <code>Vault.seek_text</code> / <code>seek_hybrid</code>. See{" "}
        <Link to="/docs/eql">EQL reference</Link>.
      </p>

      <h2 id="txn">Transactions</h2>
      <CodeBlock lang="python">
{`with db.begin_transaction() as txn:
    v = txn.vault("documents")
    v.place([1.0, 0.0], {"tag": "a"})
    v.place([0.0, 1.0], {"tag": "b"})
    # clean exit → commit; raised exception → auto-rollback`}
      </CodeBlock>
      <p>
        Transactions buffer <code>place</code> and <code>erase</code>{" "}
        calls and validate eagerly (dimension &amp; finiteness). Commit
        applies operations in order, each one WAL-appended before the
        in-memory mutation. See{" "}
        <Link to="/docs/internals/transaction-engine">Transaction engine</Link>.
      </p>

      <h2 id="embedders">Embedders</h2>
      <p>Three options, in order of preference:</p>
      <ol>
        <li><strong>Default local embedder</strong> — automatically attached on new databases. Persisted under <code>text_embedder/</code> and restored on reopen.</li>
        <li><strong>Explicit <code>LocalEmbedderConfig</code></strong> — pin model/revision/path. Rehydratable.</li>
        <li><strong>Python callable</strong> — full flexibility, but only metadata persists. Reopening without the same callable makes text-first APIs fail with <code>ConfigError</code>.</li>
      </ol>

      <h2 id="lifecycle">Persistence &amp; lifecycle</h2>
      <ul>
        <li><code>checkpoint()</code> — write manifest+segments (or snapshot) and truncate the WAL.</li>
        <li><code>compact()</code> — rebuild indexes then checkpoint.</li>
        <li><code>close()</code> — checkpoint and release locks.</li>
        <li><code>abandon()</code> — testing hook that leaves recovery work in the WAL.</li>
      </ul>
      <CodeBlock lang="python">
{`reader = elips.open("/tmp/elips-sdk", access_mode="read_only")
print(reader.vault("documents").seek_text("alpha", top=1)[0].data)`}
      </CodeBlock>
      <p>
        Read-only opens take a shared advisory lock and reject{" "}
        <code>place</code>, <code>place_document</code>, <code>erase</code>,{" "}
        <code>rebuild_index</code>, and compaction-driven mutation with{" "}
        <code>StorageError</code>.
      </p>

      <h2 id="errors">Errors</h2>
      <ul>
        <li><code>ConfigError</code> — invalid config, dimension mismatch, missing text embedder on text-first calls, or read-only open against a missing database.</li>
        <li><code>DimensionMismatch</code> / <code>InvalidVector</code> — eager-validated on every <code>place</code>.</li>
        <li><code>LockConflict</code> — another writer already holds the database.</li>
        <li><code>StorageError</code> — IO failure or mutation attempted in read-only mode.</li>
        <li><code>NotFound</code> — missing record.</li>
        <li><code>ParseError</code> — malformed EQL.</li>
      </ul>

      <h2 id="typing">Typing</h2>
      <p>
        The package ships <code>py.typed</code> and a complete{" "}
        <code>_core.pyi</code> stub, including the modern wrapper classes,
        so IDEs and type checkers see the full public API.
      </p>

      <h2 id="pitfalls">Pitfalls</h2>
      <ul>
        <li>Calling <code>seek_text</code> / <code>place_document</code> without a configured embedder raises <code>ConfigError</code> — by design, not silent fallback.</li>
        <li>Reopening a database that was created with a Python-callable embedder requires the same callable; without it text-first APIs fail.</li>
        <li><code>":memory:"</code> opens require <code>dimension &gt; 0</code> every time.</li>
        <li>Only one read-write opener at a time per database directory. Use <code>access_mode="read_only"</code> for shared-reader serving.</li>
      </ul>
    </DocsShell>
  );
}
