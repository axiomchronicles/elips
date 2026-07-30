import { createFileRoute } from "@tanstack/react-router";
import { StandalonePage } from "../components/Page";

export const Route = createFileRoute("/changelog")({
  head: () => ({
    meta: [
      { title: "Changelog — ELIPS" },
      {
        name: "description",
        content: "What shipped in each ELIPS release — features, fixes, and deferred work.",
      },
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
    v: "1.1.0",
    date: "2026-07-30",
    tag: "Latest",
    items: [
      "vacuum() / pending_removals() on Vault and Database: reclaim index space from deleted records. HNSW tombstones are now bounded — search() scales ef by the live/dead ratio; graph compacts automatically when tombstones exceed GraphParams::compaction_ratio (default 0.2).",
      "Filtered ANN search re-probes adaptively: seek() widens the fetch beam 4× per round until top results are found or the vault is exhausted, replacing the fixed over-fetch that returned short sets for selective filters.",
      "In-process reader/writer locking: each Vault now owns a std::shared_mutex; ElipsInstance owns a mutex over the registry, WAL handle, and lifecycle flags. Transaction::commit() holds the instance lock for the whole batch. Vault::records() now returns a copy for thread safety.",
      "Sealed vaults: close() marks every vault sealed; writes after close() throw StorageError instead of silently discarding.",
      "WAL and checkpoint writes now reach stable storage before acknowledgement: fdatasync on Linux, F_FULLFSYNC on macOS — acknowledged writes now survive OS crash and power loss, not just process crash.",
      "Transactions are atomic under I/O and read-only failure: commit() pre-checks writability, records an undo log, and restores prior state in reverse on failure. WAL txn_begin/txn_commit markers discard unterminated windows on replay.",
      "Length-prefixed reads bounded before allocating: WAL/snapshot length fields validated against remaining bytes before any allocation.",
      "WAL replay is O(n): eliminated the O(n²) tail-copy; 4× data now scales linearly.",
      "GPU suballocator leaks fixed: remainder of reused blocks now returned to the free list; frees coalesce adjacent same-root spans; bytes_available() reports reachable bytes.",
      "GPU engine no longer compiled on platforms with no backend: ELIPS_GPU_METAL defaults to Apple only.",
      "ELIPS_SANITIZE=thread|address CMake option; CI runs ThreadSanitizer, ASan+UBSan, and a no-GPU Linux build.",
      "Parser robustness and fuzz testing (elips_fuzz_wal); 895k ASan+UBSan executions with no findings.",
    ],
  },
  {
    v: "1.0.0",
    date: "2024-08-01",
    tag: "Baseline",
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
    <StandalonePage
      eyebrow="Project"
      title="Changelog"
      lede="Versions are tagged in the repository. This page mirrors the project's release notes and the documented roadmap."
    >
      <div className="not-prose space-y-10">
        {ENTRIES.map((e) => (
          <article
            key={e.v}
            className="grid grid-cols-12 gap-6 pt-6 hairline-t first:border-t-0 first:pt-0"
          >
            <div className="col-span-12 md:col-span-3">
              <div className="eyebrow text-primary">{e.tag}</div>
              <div className="text-ink text-[22px] mt-1" style={{ letterSpacing: "-0.01em" }}>
                {e.v}
              </div>
              <div className="text-muted text-[13px] mt-1">{e.date}</div>
            </div>
            <ul className="col-span-12 md:col-span-9 prose">
              {e.items.map((it) => (
                <li key={it}>{it}</li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </StandalonePage>
  );
}
