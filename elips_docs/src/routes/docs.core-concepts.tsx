import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";
import { SketchCard, ObjectModelDiagram } from "../components/SketchDiagram";

export const Route = createFileRoute("/docs/core-concepts")({
  head: () => ({
    meta: [
      { title: "Core concepts — ELIPS Docs" },
      {
        name: "description",
        content:
          "Databases, collections, records, document attachments, chunks, and embedding lineage — the mental model behind ELIPS.",
      },
      { property: "og:title", content: "Core concepts — ELIPS" },
      { property: "og:description", content: "Databases, collections, records, documents, and lineage." },
      { property: "og:url", content: "/docs/core-concepts" },
    ],
    links: [{ rel: "canonical", href: "/docs/core-concepts" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Concepts"
      title="Core concepts"
      toc={[
        { id: "engine", label: "Database" },
        { id: "vault", label: "Collection" },
        { id: "record", label: "Record" },
        { id: "lineage", label: "Document & lineage" },
        { id: "filters", label: "Filters" },
        { id: "plan", label: "Query plan" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS uses five nouns: <strong>database</strong>, <strong>collection</strong>,{" "}
        <strong>record</strong>, <strong>document</strong>, and <strong>lineage</strong>. They map
        directly to types in both the C++ and Python surfaces.
      </p>

      <SketchCard caption="The object model. Each database owns collections; each collection owns an index and a record store.">
        <ObjectModelDiagram />
      </SketchCard>

      <h2 id="engine">Database</h2>
      <p>
        <code>Database</code> (created via <code>elips.Client</code> /{" "}
        <code>elips.PersistentClient</code>) is the database handle. It owns the configured text
        embedder, the collection registry, the WAL, and any optional GPU device state. Create one
        with <code>elips.Client(dimension=384)</code> for an in-memory database or{" "}
        <code>elips.PersistentClient(path, dimension=384)</code> on disk; the handle is
        non-copyable.
      </p>

      <h2 id="vault">Collection</h2>
      <p>
        A collection is a named partition. Inside a collection, ids are unique, and records share
        the database's dimension and metric. A collection owns exactly one index instance (a
        FAISS-style <code>Index</code>) plus its metadata index and planner.
      </p>
      <CodeBlock lang="python">
        {`docs = client.get_or_create_collection("documents")
faces = client.get_or_create_collection("faces")`}
      </CodeBlock>

      <h2 id="record">Record</h2>
      <p>
        The record is the unit of storage. Every field except <code>key</code> and{" "}
        <code>vector</code> is optional:
      </p>
      <CodeBlock lang="cpp">
        {`struct Record {
  RecordID key;
  Vector   vector;
  Payload  meta;                                    // typed metadata
  std::optional<DocumentAttachment> document;       // raw text + uri + mime
  std::optional<ChunkInfo>          chunk;          // document key + position
  std::optional<EmbeddingLineage>   lineage;        // provider · model · revision
};`}
      </CodeBlock>

      <h2 id="lineage">Documents &amp; lineage</h2>
      <p>
        <code>DocumentAttachment</code> stores the original text and optional URI/MIME.{" "}
        <code>ChunkInfo</code> records where the record sits inside a larger document.{" "}
        <code>EmbeddingLineage</code> records which embedder produced the vector, so future
        migrations or audits can reason about provenance. All three are persisted by{" "}
        <code>WAL::insert_ex</code> and replayed on recovery.
      </p>

      <h2 id="filters">Filters</h2>
      <p>
        Metadata filters are expressed as a <code>where</code> dict with Chroma-style operators. The
        same predicate tree is built from this dict and by the EQL parser; the executor cannot tell
        which produced it. Equality and set-membership predicates accelerate through{" "}
        <code>MetadataIndex</code>; other comparators evaluate during the scan.
      </p>
      <CodeBlock lang="python">
        {`where = {"kind": "design", "year": {"$gt": 2023}}
rows = docs.get(where=where)`}
      </CodeBlock>

      <h2 id="plan">Query plan</h2>
      <p>
        Every vector or hybrid query first goes through the collection planner (<code>
          Collection::plan_query()
        </code>). The result is a <code>QueryPlan</code> that names the strategy (
        <code>ann_index</code>, <code>exact_candidates</code>, <code>full_scan</code>,{" "}
        <code>text_probe</code>, or <code>hybrid_fusion</code>), whether <code>MetadataIndex</code>{" "}
        was used, and the candidate count when relevant. The plan is exposed in both surfaces via{" "}
        <code>explain_query</code> / <code>Collection.explain()</code> — see{" "}
        <Link to="/docs/advanced">Advanced patterns</Link>.
      </p>
    </DocsShell>
  );
}
