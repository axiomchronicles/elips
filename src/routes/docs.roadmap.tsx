import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";

export const Route = createFileRoute("/docs/roadmap")({
  head: () => ({
    meta: [
      { title: "Roadmap — ELIPS Docs" },
      {
        name: "description",
        content:
          "What ELIPS v1.0 ships today and what is intentionally deferred, with the v1.0 hook that keeps each future capability additive.",
      },
      { property: "og:title", content: "Roadmap — ELIPS" },
      {
        property: "og:description",
        content:
          "Shipped vs deferred — clearly separated, with v1.0 hooks for each future capability.",
      },
      { property: "og:url", content: "/docs/roadmap" },
    ],
    links: [{ rel: "canonical", href: "/docs/roadmap" }],
  }),
  component: Page,
});

const DEFERRED: Array<[string, string]> = [
  ["Per-segment indexes + compaction", "IndexPort; snapshot/WAL split per vault"],
  ["Full MVCC version chains / Snapshot Isolation", "Single-writer model + transaction buffer"],
  ["Quantized indexes (PQ / OPQ / SQ), DiskANN", "IndexPort + make_index factory"],
  ["AVX2 / AVX-512 distance kernels", "Function-pointer dispatch seam in Metrics"],
  ["Columnar metadata, attribute B-trees, inverted / bloom", "Filter over Payload today"],
  ["Multi-reader shared locks beyond current model", "LockManager seam"],
  ["Multi-node replication / sharding", "WAL is a logical, streamable log"],
  ["Cloud object-storage adapters (S3 / GCS / Azure)", "StoragePort adapter pattern"],
  ["NumPy zero-copy ingestion, async/streaming C++ APIs", "Binding + SDK extension points"],
  ["Cross-platform snapshot encoding (little-endian)", "Centralized Serialization"],
];

function Page() {
  return (
    <DocsShell
      eyebrow="Project"
      title="Roadmap"
      toc={[
        { id: "shipped", label: "Shipped in v1.0" },
        { id: "deferred", label: "Deferred (with v1.0 hooks)" },
        { id: "limitations", label: "Known v1.0 limitations" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS v1.0 is a vertically complete embedded engine. The list of deferred items is
        intentional: each future capability has a v1.0 hook so it can land additively, without
        breaking the surface users build against today.
      </p>

      <h2 id="shipped">Shipped in v1.0</h2>
      <ul>
        <li>C++23 core with hexagonal layering and explicit C++ Core Guidelines alignment.</li>
        <li>Metrics — cosine, euclidean, dot product — with NEON SIMD plus scalar dispatch.</li>
        <li>
          Indexes — <code>HierarchicalGraphIndex</code> (HNSW) and <code>ExactIndex</code> — behind
          a single <code>IndexPort</code>.
        </li>
        <li>
          Durable storage: <code>IDENTITY</code>, atomic snapshot, record-based WAL with CRC32C,
          deterministic crash recovery.
        </li>
        <li>
          Single-writer advisory file locking; shared read-only opens for multi-reader serving.
        </li>
        <li>
          Document-aware records — <code>DocumentAttachment</code>, <code>ChunkInfo</code>,{" "}
          <code>EmbeddingLineage</code> — persisted through <code>WAL::insert_ex</code>.
        </li>
        <li>
          Native <code>place_document</code>, <code>seek_text</code>, <code>seek_hybrid</code>,{" "}
          <code>explain_seek</code>.
        </li>
        <li>Built-in local text embedder with automatic default provisioning for new databases.</li>
        <li>
          <code>MetadataIndex</code> acceleration for equality and set-membership filters.
        </li>
        <li>
          Segmented persistence with <code>elips.manifest</code> plus per-vault segment files;{" "}
          <code>compact()</code> rebuilds indexes and rewrites segments.
        </li>
        <li>
          Dynamic metadata and the <code>Filter</code> predicate engine, shared by the SDK builders
          and the EQL parser.
        </li>
        <li>Atomic batched transactions with eager validation and RAII rollback.</li>
        <li>
          EQL — lexer, recursive-descent parser, AST, executor — covering <code>seek</code> /{" "}
          <code>fetch</code> / <code>scan</code> / <code>place</code> / <code>erase</code>.
        </li>
        <li>
          <code>elips</code> CLI; PyBind11 Python bindings with <code>py.typed</code>; benchmark
          suite.
        </li>
        <li>
          Optional GPU index family behind <code>GpuPort</code> with a documented backend fallback
          chain.
        </li>
      </ul>

      <h2 id="deferred">Deferred (with v1.0 hooks)</h2>
      <div className="not-prose my-4 rounded-lg border border-hairline overflow-hidden bg-surface">
        <table className="w-full text-[14px]">
          <thead className="bg-canvas-soft">
            <tr>
              <th className="text-left px-4 py-2 font-semibold text-ink">Future capability</th>
              <th className="text-left px-4 py-2 font-semibold text-ink">v1.0 hook</th>
            </tr>
          </thead>
          <tbody>
            {DEFERRED.map(([cap, hook]) => (
              <tr key={cap} className="border-t border-hairline align-top">
                <td className="px-4 py-3 text-ink">{cap}</td>
                <td className="px-4 py-3 text-body font-mono text-[12.5px]">{hook}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        These items are <strong>not</strong> in v1.0 today. Do not treat them as available
        behaviour; they are recorded here so the design seams that make them additive stay visible
        to contributors.
      </p>

      <h2 id="limitations">Known v1.0 limitations</h2>
      <ul>
        <li>Snapshot serialization uses native byte order — single-machine use only.</li>
        <li>Checkpoints rewrite the whole snapshot (O(N)).</li>
        <li>
          Filtered ANN search post-filters an over-fetched candidate set rather than expanding{" "}
          <code>ef</code> adaptively.
        </li>
      </ul>

      <p className="text-muted text-[13px] mt-8">
        See <Link to="/docs/design-decisions">design decisions</Link> for the ADRs that capture why
        these trade-offs were made.
      </p>
    </DocsShell>
  );
}
