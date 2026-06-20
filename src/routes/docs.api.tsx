import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/api")({
  head: () => ({
    meta: [
      { title: "API reference — ELIPS Docs" },
      { name: "description", content: "The Python and C++ API reference for ELIPS — Database, Vault, Config, Filter, QueryPlan, and the modern wrapper." },
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
    <DocsShell eyebrow="Reference" title="API reference" toc={[
      { id: "open", label: "open / connect" },
      { id: "engine", label: "Engine / Database" },
      { id: "vault", label: "Vault / Arena" },
      { id: "filter", label: "Filter" },
      { id: "plan", label: "QueryPlan" },
      { id: "types", label: "Types" },
      { id: "errors", label: "Errors" },
    ]}>
      <p className="text-[18px] text-ink">
        ELIPS exposes two Python surfaces over the same C++ core. The
        low-level bindings mirror C++ 1:1; the modern wrapper adds a typed,
        text-first surface around <code>RecordInput</code>, <code>Row</code>, and{" "}
        <code>Hit</code>.
      </p>

      <h2 id="open">open / connect</h2>
      <Sig
        name="elips.open"
        py='elips.open(path: str, *, dimension: int, metric: str = "cosine", index: str = "graph", access_mode: str = "read_write", use_default_text_embedder: bool = True) -> Database'
        cpp='elips::open(std::string path, elips::Config cfg) -> std::unique_ptr<ElipsInstance>'
      />
      <Sig
        name="elips.connect (modern)"
        py='elips.connect(path: str, *, dimension: int, metric: str = "cosine", embedder: Callable | None = None) -> Engine'
      />
      <p>
        <code>open</code> takes durable identity (dimension, metric, index)
        plus runtime config (access mode, embedders). <code>connect</code> wraps{" "}
        <code>open</code> with the modern <code>Engine</code>.
      </p>

      <h2 id="engine">Engine / Database</h2>
      <Sig name="Database.vault" py='db.vault(name: str) -> Vault' cpp='ElipsInstance::vault(std::string name) -> Vault&' />
      <Sig name="Database.query" py='db.query(eql: str, *, bindings: dict | None = None) -> list[Row]' cpp='ElipsInstance::query(std::string eql, std::map<std::string, Vector> bindings) -> std::vector<Row>' />
      <Sig name="Database.begin_transaction" py='db.begin_transaction() -> Transaction' cpp='ElipsInstance::begin_transaction() -> Transaction' />
      <Sig name="Database.checkpoint / compact / close" py='db.checkpoint(); db.compact(); db.close()' cpp='db->checkpoint(); db->compact(); db->close();' />

      <h2 id="vault">Vault / Arena</h2>
      <CodeBlock lang="python">
{`# low-level
docs.place(vector, payload, *, document=None, chunk=None, lineage=None)
docs.place_document(text, payload, *, lineage=None)
docs.erase(record_id)
docs.fetch(record_id) -> Record | None
docs.scan(*, where=None, offset=0, limit=None) -> list[Record]
docs.seek(vector, top=10, *, where=None, threshold=None) -> list[SearchResult]
docs.seek_text(text, top=10, *, where=None) -> list[SearchResult]
docs.seek_hybrid(vector, text, top=10, *, where=None) -> list[SearchResult]
docs.explain_seek(vector, top=10, *, where=None, has_text_component=False) -> QueryPlan
docs.rebuild_index()`}
      </CodeBlock>
      <CodeBlock lang="python">
{`# modern wrapper (Arena)
arena.write_many(records: list[RecordInput | dict]) -> list[RecordKey]
arena.pull(keys: list[RecordKey], *, include_vectors: bool = False) -> list[Row]
arena.probe(vector, top=10, *, where=None) -> list[Hit]
arena.probe_text(text, top=10, *, where=None) -> list[Hit]
arena.probe_hybrid(vector, text, top=10, *, where=None) -> list[Hit]
arena.ingest(texts=[...], meta=[...])  # legacy column shape`}
      </CodeBlock>

      <h2 id="filter">Filter</h2>
      <CodeBlock lang="python">
{`f = (elips.Filter()
     .field("kind").equals("design")
     .and_().field("year").greater_than(2023)
     .or_().field("country").is_in(["US", "GB", "CA"]))`}
      </CodeBlock>
      <p>
        Comparators: <code>equals</code>, <code>not_equals</code>,{" "}
        <code>less_than</code>, <code>less_or_equal</code>,{" "}
        <code>greater_than</code>, <code>greater_or_equal</code>,{" "}
        <code>is_in</code>, <code>contains</code>. Boolean combinators:{" "}
        <code>and_</code>, <code>or_</code>, <code>not_</code>.
      </p>

      <h2 id="plan">QueryPlan</h2>
      <table>
        <thead><tr><th>Field</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td><code>strategy</code></td><td><code>ann_index</code> · <code>exact_candidates</code> · <code>full_scan</code> · <code>text_probe</code> · <code>hybrid_fusion</code></td></tr>
          <tr><td><code>metadata_accelerated</code></td><td>Whether the planner narrowed via <code>MetadataIndex</code>.</td></tr>
          <tr><td><code>candidate_count</code></td><td>Candidate set size when narrowed.</td></tr>
          <tr><td><code>has_text_component</code></td><td>Whether the plan includes a text stage.</td></tr>
        </tbody>
      </table>

      <h2 id="types">Types</h2>
      <ul>
        <li><code>DocumentAttachment(text, uri=None, mime=None, attributes={})</code></li>
        <li><code>ChunkInfo(document_key, position, length, attributes={})</code></li>
        <li><code>EmbeddingLineage(provider, model, revision, attributes={})</code></li>
        <li><code>SearchResult(id, distance, payload, document?, chunk?, lineage?)</code></li>
        <li><code>RecordInput(text=..., vector=..., meta=..., document=..., chunk=..., lineage=...)</code></li>
        <li><code>Hit(key, distance, text, meta, vector?, document?)</code></li>
      </ul>

      <h2 id="errors">Errors</h2>
      <p>All runtime errors derive from <code>ElipsError</code>:</p>
      <ul>
        <li><code>DimensionMismatch</code> — vector width disagrees with identity.</li>
        <li><code>InvalidVector</code> — NaN, empty, or otherwise malformed input.</li>
        <li><code>ConfigError</code> — incompatible config or missing embedder on reopen.</li>
        <li><code>StorageError</code> — write attempted on a read-only handle, or persistence layer failure.</li>
        <li><code>LockConflict</code> — another writer holds the database.</li>
        <li><code>NotFound</code> — record id absent.</li>
      </ul>
    </DocsShell>
  );
}
