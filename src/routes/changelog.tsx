import { createFileRoute } from "@tanstack/react-router";
import { StandalonePage } from "../components/Page";

export const Route = createFileRoute("/changelog")({
  head: () => ({
    meta: [
      { title: "Changelog — ELIPS" },
      { name: "description", content: "What shipped in each ELIPS release — features, fixes, and deferred work." },
      { property: "og:title", content: "Changelog — ELIPS" },
      { property: "og:description", content: "ELIPS release notes." },
      { property: "og:url", content: "/changelog" },
    ],
    links: [{ rel: "canonical", href: "/changelog" }],
  }),
  component: Page,
});

const ENTRIES = [
  {
    v: "1.0.0",
    date: "Current",
    tag: "Stable",
    items: [
      "C++23 core with hexagonal layering and full Core Guidelines compliance.",
      "HierarchicalGraphIndex (HNSW) and ExactIndex behind a single IndexPort.",
      "First-class DocumentAttachment, ChunkInfo, and EmbeddingLineage.",
      "Native place_document, seek_text, seek_hybrid, and explain_seek.",
      "Built-in local text embedder with automatic default provisioning for new databases.",
      "MetadataIndex acceleration for equality and set-membership filters.",
      "Segmented persistence with elips.manifest plus per-vault segment files.",
      "compact() rebuilds indexes and rewrites the segment set.",
      "Shared read-only mode with advisory locks.",
      "WAL crash recovery, snapshot compatibility, typed filters, EQL, Python bindings.",
      "Optional GPU index family behind GpuPort.",
    ],
  },
  {
    v: "Deferred",
    date: "Future",
    tag: "Roadmap",
    items: [
      "Per-segment indexes plus compaction (hooked through IndexPort).",
      "Full MVCC version chains and snapshot isolation.",
      "Quantised indexes (PQ / OPQ / SQ) and DiskANN.",
      "AVX2 / AVX-512 distance kernels.",
      "Columnar metadata, attribute B-trees, inverted / bloom indexes.",
      "Cloud object-storage adapters (S3 / GCS / Azure) behind StoragePort.",
      "NumPy zero-copy ingestion and async/streaming C++ APIs.",
    ],
  },
];

function Page() {
  return (
    <StandalonePage eyebrow="Project" title="Changelog" lede="Versions are tagged in the repository. This page mirrors the project's release notes and the documented roadmap.">
      <div className="not-prose space-y-10">
        {ENTRIES.map((e) => (
          <article key={e.v} className="grid grid-cols-12 gap-6 pt-6 hairline-t first:border-t-0 first:pt-0">
            <div className="col-span-12 md:col-span-3">
              <div className="eyebrow text-primary">{e.tag}</div>
              <div className="text-ink text-[22px] mt-1" style={{ letterSpacing: "-0.01em" }}>{e.v}</div>
              <div className="text-muted text-[13px] mt-1">{e.date}</div>
            </div>
            <ul className="col-span-12 md:col-span-9 prose">
              {e.items.map((it) => <li key={it}>{it}</li>)}
            </ul>
          </article>
        ))}
      </div>
    </StandalonePage>
  );
}
