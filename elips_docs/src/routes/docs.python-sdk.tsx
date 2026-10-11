import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python-sdk")({
  head: () => ({
    meta: [
      { title: "Python SDK — ELIPS Docs" },
      {
        name: "description",
        content:
          "The ELIPS Python surface — the Chroma-style Client/Collection API, the FAISS-style Index API, embedders, EQL, transactions, and read-only serving.",
      },
      { property: "og:title", content: "Python SDK — ELIPS" },
      {
        property: "og:description",
        content:
          "Client, Collection, the FAISS Index API, Config, Filter, EQL, embedders, and persistence.",
      },
      { property: "og:url", content: "/docs/python-sdk" },
    ],
    links: [{ rel: "canonical", href: "/docs/python-sdk" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Python SDK"
      toc={[
        { id: "install", label: "Install & build" },
        { id: "two-layers", label: "Two surfaces" },
        { id: "client", label: "Client API" },
        { id: "collection", label: "Collection API" },
        { id: "index", label: "FAISS Index API" },
        { id: "config", label: "Config" },
        { id: "query", label: "Query & planner" },
        { id: "eql", label: "EQL from Python" },
        { id: "txn", label: "Transactions" },
        { id: "embedders", label: "Embedders" },
        { id: "lifecycle", label: "Persistence & lifecycle" },
        { id: "errors", label: "Errors" },
        { id: "pitfalls", label: "Pitfalls" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS ships two Python surfaces over the same C++ core. The high-level surface is{" "}
        <strong>Chroma-style</strong>: <code>Client</code> / <code>PersistentClient</code> hand out{" "}
        <code>Collection</code> objects whose methods are{" "}
        <code>add/upsert/get/update/delete/query</code>. Alongside it is a{" "}
        <strong>FAISS-style</strong> <code>Index</code> API (<code>IndexFlat</code>,{" "}
        <code>IndexIVFFlat</code>, <code>bf_knn</code>, …) for raw array search. Both speak to the
        same vectors, the same WAL, and the same planner.
      </p>

      <h2 id="install">Install &amp; build</h2>
      <p>
        The bindings are built from the repository — there is no PyPI wheel yet. Build the extension
        and put the package on <code>PYTHONPATH</code>:
      </p>
      <CodeBlock lang="bash">
        {`cmake -S . -B build -G Ninja -DELIPS_ENABLE_PYTHON=ON
cmake --build build --target elips_pymodule
export PYTHONPATH=$PWD/bindings/python`}
      </CodeBlock>

      <h2 id="two-layers">Two surfaces, one core</h2>
      <ul>
        <li>
          <strong>Chroma-style</strong>: <code>Client()</code>, <code>PersistentClient()</code>,{" "}
          <code>Collection</code>, <code>RecordInput</code>, <code>Row</code>, <code>Hit</code>.
          Pick this for typed, text-first ingestion and retrieval. The pre-1.1 openers{" "}
          <code>connect()</code> / <code>open()</code> / <code>Engine</code> and the collection
          verbs <code>write</code>/<code>probe</code>/<code>seek</code> remain as deprecated shims.
        </li>
        <li>
          <strong>FAISS-style</strong>: <code>IndexFlat</code>, <code>IndexIVFFlat</code>,{" "}
          <code>IndexIVFPQ</code>, <code>IndexHNSW</code>, <code>index_factory</code>, and{" "}
          <code>bf_knn</code>. Pick this when you want raw NumPy-array search with FAISS semantics
          and no record store.
        </li>
      </ul>

      <h2 id="client">Client API</h2>
      <CodeBlock lang="python">
        {`import elips

client = elips.PersistentClient("/tmp/elips-sdk", dimension=128, metric="cosine")
docs = client.get_or_create_collection("documents")

docs.add(
    ids=["a", "b"],
    documents=["alpha design note", "beta runbook"],
    metadatas=[{"kind": "design"}, {"kind": "ops"}],
)

hits = docs.query(query_texts="alpha", n_results=2)
print(hits[0].text, hits[0].distance)
client.close()`}
      </CodeBlock>
      <ul>
        <li>
          <code>Client(**settings)</code> — ephemeral in-memory;{" "}
          <code>PersistentClient(path, **settings)</code> — on disk. See{" "}
          <Link to="/docs/python/connect">Opening a database</Link>.
        </li>
        <li>
          <code>create_collection</code> / <code>get_collection</code> /{" "}
          <code>get_or_create_collection</code> / <code>list_collections</code> — Chroma collection
          management. <code>delete_collection</code> raises (ELIPS cannot drop a collection yet).
        </li>
        <li>
          <code>checkpoint()</code>, <code>compact()</code>, <code>vacuum()</code>,{" "}
          <code>pending_writes()</code>, <code>query_eql()</code>, <code>close()</code>, and the{" "}
          <code>with</code> context manager.
        </li>
        <li>
          <code>client.native</code> — the low-level <code>Database</code> handle; the explicit
          escape hatch.
        </li>
      </ul>

      <h2 id="collection">Collection API</h2>
      <CodeBlock lang="python">
        {`# Column form (Chroma)
docs.add(
    ids=["x"],
    embeddings=[[1.0, 0.0]],
    documents=["gamma appendix"],
    metadatas=[{"kind": "appendix"}],
)

# Single record
docs.add(embedding=[0.0, 1.0], metadata={"kind": "ops"}, id="y")

# Read, update, delete
rows = docs.get(ids=["x"], include=["embeddings"])
docs.update(ids=["x"], metadatas=[{"kind": "appendix", "rev": 2}])
docs.delete(ids=["y"])

print(docs.count(), docs.peek(5))`}
      </CodeBlock>
      <p>
        The canonical methods are <code>add</code>, <code>upsert</code>, <code>get</code>,{" "}
        <code>update</code>, <code>delete</code>, <code>query</code>, <code>count</code>, and{" "}
        <code>peek</code>. <code>add</code> has upsert semantics (re-adding an id overwrites it).
        ELIPS adds <code>explain()</code>, <code>health()</code>, <code>freeze()</code>,{" "}
        <code>vacuum()</code>, <code>rebuild()</code>, and <code>compress()</code> (alias{" "}
        <code>train</code>). Reads return <code>Row</code> / <code>Hit</code> dataclasses with{" "}
        <code>.key</code>, <code>.meta</code>, <code>.text</code>, and (on hits){" "}
        <code>.distance</code>. See <Link to="/docs/python/arena">Collection</Link>.
      </p>

      <h2 id="index">FAISS Index API</h2>
      <p>
        For raw array search without the record store, ELIPS exposes a FAISS-style index surface:
      </p>
      <CodeBlock lang="python">
        {`import numpy as np, elips

xb = np.random.rand(100_000, 128).astype("float32")
xq = np.random.rand(1_000, 128).astype("float32")

index = elips.IndexIVFFlat(128, 316, "l2")
index.train(xb)
index.add(xb)
index.nprobe = 128
D, I = index.search(xq, 10)      # distances, int64 labels

# Exact brute force, no build — CPU or Apple GPU
D, I = elips.bf_knn(xq, xb, 10, metric="l2", device="metal")`}
      </CodeBlock>
      <ul>
        <li>
          Constructors: <code>IndexFlat(d, metric)</code>,{" "}
          <code>IndexIVFFlat(d, nlist, metric)</code>,{" "}
          <code>IndexIVFPQ(d, nlist, M, metric)</code>,{" "}
          <code>IndexIVFScalarQuantizer(d, nlist, metric)</code>,{" "}
          <code>IndexHNSW(d, M, metric)</code>, and <code>index_factory(d, spec, metric)</code>.
        </li>
        <li>
          Methods: <code>train</code>, <code>add</code>, <code>add_with_ids</code>,{" "}
          <code>search(x, k)</code>, <code>reset</code>; properties <code>ntotal</code>,{" "}
          <code>d</code>, <code>is_trained</code>, and the settable <code>nprobe</code>.
        </li>
        <li>
          <code>bf_knn</code> is present only in GPU-enabled builds; <code>device="metal"</code>{" "}
          runs the Apple-GPU exact path. See <Link to="/docs/benchmarks">benchmarks</Link>.
        </li>
      </ul>

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
client = elips.connect_with_config("/tmp/elips-quickstart", config)`}
      </CodeBlock>
      <ul>
        <li>
          <code>segmented_storage(True)</code> — default. Writes <code>elips.manifest</code> +
          per-collection segment files.
        </li>
        <li>
          <code>metadata_acceleration(True)</code> — equality and set-membership filters narrow
          candidates through the metadata filter index.
        </li>
        <li>
          <code>auto_text_embedder(True)</code> — provisions the built-in local embedder for new
          databases.
        </li>
        <li>
          <code>local_text_embedder(...)</code> / <code>text_embedder(callable, ...)</code> — pin a
          rehydratable local embedder, or attach a Python callable (metadata-only persistence).
        </li>
      </ul>

      <h2 id="query">Query &amp; planner</h2>
      <CodeBlock lang="python">
        {`# Vector
hits = docs.query(query_embeddings=[1.0, 0.0], n_results=2)

# Text-first
hits = docs.query(query_texts="alpha", n_results=2)

# Hybrid (vector + lexical overlap from attached documents)
hits = docs.query(
    query_embeddings=[0.0, 1.0],
    query_texts="alpha",
    n_results=2,
    lexical_weight=0.35,
)

# Inspect the plan
where = elips.Filter().field("kind").equals("design")
plan = docs.explain([1.0, 0.0], top=1, where=where, has_text_component=True)
print(plan.strategy.name, plan.metadata_accelerated, plan.candidate_count)`}
      </CodeBlock>
      <p>
        The planner always emits one of <code>ann_index</code>, <code>exact_candidates</code>,{" "}
        <code>full_scan</code>, <code>text_probe</code>, <code>hybrid_fusion</code> — exposed
        identically here and in EQL.
      </p>

      <h2 id="eql">EQL from Python</h2>
      <CodeBlock lang="python">
        {`rows = client.query_eql(
    'search in documents nearest $q k 5 where kind = "design" yield',
    {"q": [1.0, 0.0]},
)`}
      </CodeBlock>
      <p>
        EQL is a small lowercase DSL — bindings use <code>$name</code>. Text-first retrieval is not
        exposed through EQL; use <code>Collection.query(query_texts=...)</code>. See the{" "}
        <Link to="/docs/eql">EQL reference</Link>.
      </p>

      <h2 id="txn">Transactions</h2>
      <CodeBlock lang="python">
        {`with client.native.begin_transaction() as txn:
    v = txn.vault("documents")
    v.place([1.0, 0.0], {"tag": "a"})
    v.place([0.0, 1.0], {"tag": "b"})
    # clean exit → commit; raised exception → auto-rollback`}
      </CodeBlock>
      <p>
        Transactions live on the native handle (<code>client.native.begin_transaction()</code>) and
        validate eagerly (dimension &amp; finiteness). Commit applies operations in order, each
        WAL-appended before the in-memory mutation. See{" "}
        <Link to="/docs/internals/transaction-engine">Transaction engine</Link>.
      </p>

      <h2 id="embedders">Embedders</h2>
      <ol>
        <li>
          <strong>Default local embedder</strong> — automatically attached on new databases.
          Persisted under <code>text_embedder/</code> and restored on reopen.
        </li>
        <li>
          <strong>
            Explicit <code>LocalEmbedderConfig</code>
          </strong>{" "}
          — pin model/revision/path. Rehydratable.
        </li>
        <li>
          <strong>Python callable</strong> — full flexibility, but only metadata persists. Reopening
          without the same callable makes text-first APIs fail with <code>ConfigError</code>.
        </li>
      </ol>

      <h2 id="lifecycle">Persistence &amp; lifecycle</h2>
      <ul>
        <li>
          <code>checkpoint()</code> — write manifest+segments (or snapshot) and truncate the WAL.
        </li>
        <li>
          <code>compact()</code> — rebuild indexes then checkpoint.
        </li>
        <li>
          <code>vacuum()</code> — reclaim index space held by deleted records.
        </li>
        <li>
          <code>close()</code> — checkpoint and release locks.
        </li>
      </ul>
      <CodeBlock lang="python">
        {`reader = elips.PersistentClient("/tmp/elips-sdk", access_mode="read_only")
docs = reader.get_collection("documents")
print(docs.query(query_texts="alpha", n_results=1)[0].meta)`}
      </CodeBlock>
      <p>
        Read-only opens take a shared advisory lock and reject writes with{" "}
        <code>StorageError</code>.
      </p>

      <h2 id="errors">Errors</h2>
      <ul>
        <li>
          <code>ConfigError</code> — invalid config, missing text embedder on text-first calls, or
          read-only open against a missing database.
        </li>
        <li>
          <code>DimensionMismatch</code> / <code>InvalidVector</code> — eager-validated on every{" "}
          <code>add</code>.
        </li>
        <li>
          <code>CollectionNotFoundError</code> — <code>get_collection</code> on an absent name.
        </li>
        <li>
          <code>NotTrainedError</code> / <code>BackendUnavailableError</code> — untrained index use,
          or an absent GPU backend requested without <code>fallback</code>.
        </li>
        <li>
          <code>LockConflict</code> — another writer already holds the database.
        </li>
        <li>
          <code>StorageError</code> — IO failure or mutation in read-only mode;{" "}
          <code>NotFound</code> — missing record; <code>ParseError</code> — malformed EQL.
        </li>
      </ul>

      <h2 id="pitfalls">Pitfalls</h2>
      <ul>
        <li>
          Text-first calls (<code>query(query_texts=...)</code>, text-only <code>add</code>) without
          a configured embedder raise <code>ConfigError</code> — by design, not silent fallback.
        </li>
        <li>
          Reopening a database created with a Python-callable embedder requires the same callable.
        </li>
        <li>
          <code>Client()</code> and in-memory opens require <code>dimension &gt; 0</code>.
        </li>
        <li>
          Only one read-write opener at a time per directory. Use{" "}
          <code>access_mode="read_only"</code> for shared-reader serving.
        </li>
      </ul>
    </DocsShell>
  );
}
