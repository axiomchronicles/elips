import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/api")({
  head: () => ({
    meta: [
      { title: "API reference — ELIPS Docs" },
      {
        name: "description",
        content:
          "The Python and C++ API reference for ELIPS — Client, Collection, the FAISS Index API, Config, Filter, and QueryPlan.",
      },
      { property: "og:title", content: "API reference — ELIPS" },
      { property: "og:description", content: "ELIPS Python and C++ API reference." },
      { property: "og:url", content: "/docs/api" },
    ],
    links: [{ rel: "canonical", href: "/docs/api" }],
  }),
  component: Page,
});

function Sig({ name, py, cpp }: { name: string; py: string; cpp?: string }) {
  return (
    <div className="not-prose my-5 rounded-lg border border-hairline overflow-hidden">
      <div className="px-5 py-3 border-b border-hairline bg-canvas-soft flex items-baseline justify-between">
        <code className="text-ink text-[14px] font-mono">{name}</code>
      </div>
      <div className="p-5 space-y-3">
        <div>
          <div className="eyebrow mb-1">Python</div>
          <code className="font-mono text-[13px] text-body">{py}</code>
        </div>
        {cpp && (
          <div>
            <div className="eyebrow mb-1">C++</div>
            <code className="font-mono text-[13px] text-body">{cpp}</code>
          </div>
        )}
      </div>
    </div>
  );
}

function Page() {
  return (
    <DocsShell
      eyebrow="Reference"
      title="API reference"
      toc={[
        { id: "open", label: "Client / PersistentClient" },
        { id: "engine", label: "Client / Database" },
        { id: "vault", label: "Collection" },
        { id: "index", label: "Index (FAISS)" },
        { id: "filter", label: "Filter" },
        { id: "plan", label: "QueryPlan" },
        { id: "types", label: "Types" },
        { id: "errors", label: "Errors" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS exposes two Python surfaces over the same C++ core. The high-level, Chroma-style
        surface is <code>Client</code> / <code>Collection</code> around <code>RecordInput</code>,{" "}
        <code>Row</code>, and <code>Hit</code>; the FAISS-style <code>Index</code> API offers raw
        array search. The native handles mirror C++ 1:1 under <code>client.native</code>.
      </p>

      <h2 id="open">Client / PersistentClient</h2>
      <Sig
        name="elips.Client / PersistentClient"
        py='elips.Client(*, dimension: int, metric: str = "cosine", index: str = "graph", ...) ; elips.PersistentClient(path: str, *, dimension: int, ...)'
        cpp="elips::open(std::string path, elips::Config cfg) -> std::unique_ptr<elips::Database>"
      />
      <Sig
        name="elips.connect (deprecated shim)"
        py='elips.connect(path: str, *, dimension: int = 0, metric: str = "cosine", index: str = "graph", embedder=None, device=None) -> Engine'
      />
      <p>
        <code>Client</code> opens an in-memory database; <code>PersistentClient</code> opens one on
        disk. Both subclass <code>Engine</code> and carry the Chroma client surface. The pre-1.1{" "}
        <code>connect</code> / <code>open</code> openers remain as deprecated shims.
      </p>

      <h2 id="engine">Client / Database</h2>
      <Sig
        name="Client.get_or_create_collection"
        py="client.get_or_create_collection(name: str, *, embedder=None) -> Collection"
        cpp="elips::Database::collection(std::string name) -> Collection&"
      />
      <Sig
        name="Client collection management"
        py="client.create_collection(name); client.get_collection(name); client.list_collections() -> list[str]"
      />
      <Sig
        name="Client.query_eql"
        py="client.query_eql(eql: str, bindings: dict | None = None) -> list[Row]"
        cpp="elips::Database::query(std::string eql, std::map<std::string, Vector> bindings) -> std::vector<Row>"
      />
      <Sig
        name="Client.checkpoint / compact / vacuum / close"
        py="client.checkpoint(); client.compact(); client.vacuum(); client.close()"
        cpp="db->checkpoint(); db->compact(); db->close();"
      />
      <p>
        Transactions live on the native handle:{" "}
        <code>client.native.begin_transaction()</code> (C++{" "}
        <code>elips::Database::begin_transaction()</code>).
      </p>

      <h2 id="vault">Collection</h2>
      <CodeBlock lang="python">
        {`# Chroma-style surface
docs.add(*, ids=None, embeddings=None, documents=None, metadatas=None) -> str | list[str]
docs.upsert(...)                      # same as add
docs.update(*, ids, embeddings=None, documents=None, metadatas=None) -> list[str]
docs.get(ids=None, *, where=None, limit=None, offset=0, include=None) -> list[Row]
docs.delete(ids=None, *, where=None) -> int
docs.query(query_embeddings=None, query_texts=None, *, n_results=10,
           where=None, include=None, max_distance=None, lexical_weight=0.25)
           -> list[Hit] | list[list[Hit]]
docs.count() -> int
docs.peek(n=10) -> list[Row]
# ELIPS extensions
docs.explain(vector, *, top=10, where=None) -> QueryPlan
docs.health() -> CollectionHealth
docs.vacuum(); docs.rebuild(); docs.freeze(); docs.compress()`}
      </CodeBlock>
      <CodeBlock lang="cpp">
        {`// C++ Collection (renamed from Vault; erase -> remove for the keyword clash)
docs.add(vector, payload, document, chunk, lineage);
docs.add_document(text, payload, lineage);
docs.remove(record_id);
docs.get(record_id) -> std::optional<Record>;
docs.scan(where, offset, limit) -> std::vector<Record>;
docs.search(vector, k, where, threshold) -> std::vector<SearchResult>;
docs.search_text(text, k, where) -> std::vector<SearchResult>;
docs.hybrid_search(vector, text, k, where) -> std::vector<SearchResult>;
docs.explain_query(vector, k, where, has_text_component) -> QueryPlan;`}
      </CodeBlock>

      <h2 id="index">Index (FAISS-style)</h2>
      <CodeBlock lang="python">
        {`index = elips.IndexIVFFlat(d=128, nlist=316, metric="l2")
index.train(xb); index.add(xb); index.nprobe = 128
D, I = index.search(xq, k=10)       # distances, int64 labels
# Constructors: IndexFlat, IndexIVFFlat, IndexIVFPQ, IndexIVFScalarQuantizer,
#               IndexHNSW, index_factory(d, spec, metric)
# Exact brute force, no build (GPU builds): elips.bf_knn(xq, xb, k, metric="l2", device="metal")`}
      </CodeBlock>

      <h2 id="filter">Filter</h2>
      <CodeBlock lang="python">
        {`f = (elips.Filter()
     .field("kind").equals("design")
     .and_(elips.Filter().field("year").gte(2023))
     .or_(elips.Filter().field("country").is_in(["US", "GB", "CA"])))

hits = docs.query(query_embeddings=[1.0, 0.0], n_results=10, where=f)`}
      </CodeBlock>
      <p>
        Comparators map to <code>eq</code>, <code>ne</code>, <code>lt</code>, <code>le</code>,{" "}
        <code>gt</code>, <code>ge</code>, <code>gte</code>, plus <code>is_in</code> and{" "}
        <code>contains</code>. Boolean combinators: <code>and_</code>, <code>or_</code>,{" "}
        <code>not_</code>. Pass the result as <code>where=</code> to{" "}
        <code>query</code>/<code>get</code>/<code>delete</code>.
      </p>

      <h2 id="plan">QueryPlan</h2>
      <table>
        <thead>
          <tr>
            <th>Field</th>
            <th>Meaning</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <code>strategy</code>
            </td>
            <td>
              <code>ann_index</code> · <code>exact_candidates</code> · <code>full_scan</code> ·{" "}
              <code>text_probe</code> · <code>hybrid_fusion</code>
            </td>
          </tr>
          <tr>
            <td>
              <code>metadata_accelerated</code>
            </td>
            <td>
              Whether the planner narrowed via the metadata filter index.
            </td>
          </tr>
          <tr>
            <td>
              <code>candidate_count</code>
            </td>
            <td>Candidate set size when narrowed.</td>
          </tr>
          <tr>
            <td>
              <code>has_text_component</code>
            </td>
            <td>Whether the plan includes a text stage.</td>
          </tr>
        </tbody>
      </table>

      <h2 id="types">Types</h2>
      <ul>
        <li>
          <code>DocumentAttachment(text, uri=None, mime=None, attributes={})</code>
        </li>
        <li>
          <code>ChunkInfo(document_key, position, length, attributes={})</code>
        </li>
        <li>
          <code>EmbeddingLineage(provider, model, revision, attributes={})</code>
        </li>
        <li>
          <code>SearchResult</code> (C++) — <code>id</code>, <code>distance</code>,{" "}
          <code>payload</code>, <code>document?</code>, <code>chunk?</code>, <code>lineage?</code>,{" "}
          <code>codec</code>.
        </li>
        <li>
          <code>
            RecordInput(vector=..., text=..., meta=..., key=..., document=..., chunk=..., lineage=...)
          </code>
        </li>
        <li>
          <code>Row(key, meta, document?, vector?, chunk?, lineage?)</code> — <code>.text</code>{" "}
          aliases <code>document.text</code>.
        </li>
        <li>
          <code>Hit(key, distance, meta, document?, vector?, chunk?, lineage?)</code>.
        </li>
        <li>
          <code>CollectionHealth(name, live, pending_removals, dimension, metric, read_only, sealed, codec, code_bytes)</code>
        </li>
      </ul>

      <h2 id="errors">Errors</h2>
      <p>
        All runtime errors derive from <code>ElipsError</code>:
      </p>
      <ul>
        <li>
          <code>DimensionMismatch</code> — vector width disagrees with identity.
        </li>
        <li>
          <code>InvalidVector</code> — NaN, empty, or otherwise malformed input.
        </li>
        <li>
          <code>ConfigError</code> — incompatible config or missing embedder on reopen.
        </li>
        <li>
          <code>StorageError</code> — write attempted on a read-only handle, or persistence layer
          failure.
        </li>
        <li>
          <code>LockConflict</code> — another writer holds the database.
        </li>
        <li>
          <code>NotFound</code> — record id absent.
        </li>
      </ul>
    </DocsShell>
  );
}
