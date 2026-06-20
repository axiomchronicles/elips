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
          "Engines, vaults, records, document attachments, chunks, and embedding lineage — the mental model behind ELIPS.",
      },
      { property: "og:title", content: "Core concepts — ELIPS" },
      { property: "og:description", content: "Engines, vaults, records, documents, and lineage." },
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
        { id: "engine", label: "Engine" },
        { id: "vault", label: "Vault" },
        { id: "record", label: "Record" },
        { id: "lineage", label: "Document & lineage" },
        { id: "filters", label: "Filters" },
        { id: "plan", label: "Query plan" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS uses five nouns: <strong>engine</strong>, <strong>vault</strong>,{" "}
        <strong>record</strong>, <strong>document</strong>, and <strong>lineage</strong>. They map
        directly to types in both the C++ and Python surfaces.
      </p>

      <SketchCard caption="The object model. Each engine owns vaults; each vault owns an index and a record store.">
        <ObjectModelDiagram />
      </SketchCard>

      <h2 id="engine">Engine</h2>
      <p>
        <code>ElipsInstance</code> (Python: <code>Database</code> / <code>Engine</code>) is the
        database handle. It owns the configured text embedder, the vault registry, the WAL, and any
        optional GPU device state. It is created by <code>open()</code> / <code>connect()</code> and
        is non-copyable.
      </p>

      <h2 id="vault">Vault</h2>
      <p>
        A vault is a named partition. Inside a vault, ids are unique, and records share the
        database's dimension and metric. A vault owns exactly one index instance (behind{" "}
        <code>IndexPort</code>) plus its metadata index and planner.
      </p>
      <CodeBlock lang="python">
        {`docs = engine.arena("documents")
faces = engine.arena("faces")`}
      </CodeBlock>

      <h2 id="record">Record</h2>
      <p>
        The record is the unit of storage. Every field except <code>id</code> and{" "}
        <code>vector</code> is optional:
      </p>
      <CodeBlock lang="cpp">
        {`struct Record {
  RecordID id;
  Vector   vector;
  Payload  payload;                                 // typed metadata
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
        <code>Filter</code> is a value-typed predicate tree. The same object is built by the fluent
        builder and by the EQL parser; the executor cannot tell which produced it. Equality and
        set-membership predicates accelerate through <code>MetadataIndex</code>; other comparators
        evaluate during the scan.
      </p>
      <CodeBlock lang="python">
        {`f = (elips.Filter()
     .field("kind").equals("design")
     .and_().field("year").greater_than(2023))`}
      </CodeBlock>

      <h2 id="plan">Query plan</h2>
      <p>
        Every vector or hybrid query first goes through <code>Vault::plan_seek()</code>. The result
        is a <code>QueryPlan</code> that names the strategy (<code>ann_index</code>,{" "}
        <code>exact_candidates</code>, <code>full_scan</code>, <code>text_probe</code>, or{" "}
        <code>hybrid_fusion</code>), whether <code>MetadataIndex</code> was used, and the candidate
        count when relevant. The plan is exposed in both surfaces — see{" "}
        <Link to="/docs/advanced">Advanced patterns</Link>.
      </p>
    </DocsShell>
  );
}
