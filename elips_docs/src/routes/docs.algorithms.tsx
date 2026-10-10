import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";
import { SketchCard, HnswLayersDiagram } from "../components/SketchDiagram";

export const Route = createFileRoute("/docs/algorithms")({
  head: () => ({
    meta: [
      { title: "Algorithms — ELIPS Docs" },
      {
        name: "description",
        content:
          "HNSW, exact search, metadata acceleration, hybrid fusion, and the GPU index family inside ELIPS.",
      },
      { property: "og:title", content: "Algorithms — ELIPS" },
      { property: "og:description", content: "How ELIPS finds nearest neighbours." },
      { property: "og:url", content: "/docs/algorithms" },
    ],
    links: [{ rel: "canonical", href: "/docs/algorithms" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Concepts"
      title="Algorithms"
      toc={[
        { id: "metrics", label: "Distance metrics" },
        { id: "hnsw", label: "HNSW (graph)" },
        { id: "exact", label: "Exact" },
        { id: "metadata", label: "Metadata acceleration" },
        { id: "hybrid", label: "Hybrid fusion" },
        { id: "gpu", label: "GPU family" },
      ]}
    >
      <p className="lede handwritten-lede">
        Indexes plug in behind the FAISS-style <code>Index</code> base. ELIPS ships a familiar
        family — <code>IndexFlat</code>, <code>IndexHNSW</code>, <code>IndexIVFFlat</code>,{" "}
        <code>IndexIVFPQ</code>, and <code>IndexIVFScalarQuantizer</code> — built directly or via{" "}
        <code>index_factory</code>. An optional GPU path lives under <code>src/gpu_engine/</code>{" "}
        behind <code>GpuBackend</code>.
      </p>

      <h2 id="metrics">Distance metrics</h2>
      <p>
        Three metrics: <code>cosine</code>, <code>euclidean</code>, and <code>dot_product</code>.
        The native FAISS surface spells these FAISS-style — <code>metric="l2"</code> is euclidean.
        Distance kernels are dispatched through a function pointer in <code>Metrics</code> so AVX2 /
        AVX-512 / NEON variants can slot in without touching call sites. By convention, smaller
        distance always means more similar.
      </p>

      <h2 id="hnsw">HNSW (graph)</h2>
      <p>
        <code>IndexHNSW</code> implements Hierarchical Navigable Small World — a multi-layer
        proximity graph that gives sub-linear ANN with strong recall. Its knobs live on{" "}
        <code>HnswParams</code>:
      </p>
      <ul>
        <li>
          <code>M</code> — outgoing neighbours per node (default 32). Higher = better recall, more
          memory.
        </li>
        <li>
          <code>ef_construction</code> — candidate pool during insertion. Higher = better graph,
          slower writes.
        </li>
        <li>
          <code>ef_search</code> — candidate pool during search. Higher = better recall, slower
          reads.
        </li>
      </ul>
      <SketchCard caption="HNSW — sparse hubs at the top, full population at layer 0. Search starts at a hub and descends greedily, refining with ef_search at each step.">
        <HnswLayersDiagram />
      </SketchCard>

      <h2 id="exact">Exact</h2>
      <p>
        <code>IndexFlat</code> performs a brute-force scan. Use it for small collections,
        ground-truth measurement when tuning recall, and unit tests. It implements the same{" "}
        <code>Index</code> contract, so switching is a configuration change. The IVF variants —{" "}
        <code>IndexIVFFlat</code>, <code>IndexIVFPQ</code>, and{" "}
        <code>IndexIVFScalarQuantizer</code> — add an inverted-file coarse quantiser on top, tuned by{" "}
        <code>nlist</code> at build time and <code>nprobe</code> at query time.
      </p>

      <h2 id="metadata">Metadata acceleration</h2>
      <p>
        <code>MetadataIndex</code> is an exact-match accelerator for equality and set-membership
        predicates. The planner consults it first; if it produces a sufficiently small candidate
        set, the executor switches strategy to <code>exact_candidates</code> over the narrow set
        instead of running ANN over the full population.
      </p>
      <CodeBlock lang="python">
        {`plan = docs.explain(
    [1.0, 0.0],
    n_results=5,
    where={"kind": "design"},
)
print(plan.strategy, plan.metadata_accelerated, plan.candidate_count)`}
      </CodeBlock>

      <h2 id="hybrid">Hybrid fusion</h2>
      <p>
        Passing both <code>query_embeddings</code> and <code>query_texts</code> to{" "}
        <code>collection.query()</code> combines vector distance with lexical overlap on attached{" "}
        <code>DocumentAttachment</code> text (tune the blend with <code>lexical_weight</code>). The
        planner emits the <code>hybrid_fusion</code> strategy; fusion is score-level, not
        retrieval-level, so a record only needs to surface in one stage to be considered.
      </p>

      <h2 id="gpu">GPU family</h2>
      <p>
        When built with <code>-DELIPS_GPU_ENABLED=ON</code>, ELIPS accelerates the distance and
        top-k stage on the GPU behind <code>GpuBackend</code>. The exact brute-force path is exposed
        directly as <code>elips.bf_knn(xq, xb, k, device=&quot;metal&quot;)</code> — no index build.
        Metal is the implemented path (Beta, validated on Apple M4) on Apple Silicon; CUDA and HIP/ROCm are experimental. Selection
        runs through <code>BackendSelector</code> at startup; domain code never sees a backend type.
        For the full engine architecture, see <Link to="/docs/gpu-engine">GPU engine</Link>.
      </p>
    </DocsShell>
  );
}
