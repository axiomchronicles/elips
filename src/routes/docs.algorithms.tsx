import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";
import { SketchCard, HnswLayersDiagram } from "../components/SketchDiagram";

export const Route = createFileRoute("/docs/algorithms")({
  head: () => ({
    meta: [
      { title: "Algorithms — ELIPS Docs" },
      { name: "description", content: "HNSW, exact search, metadata acceleration, hybrid fusion, and the GPU index family inside ELIPS." },
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
    <DocsShell eyebrow="Concepts" title="Algorithms" toc={[
      { id: "metrics", label: "Distance metrics" },
      { id: "hnsw", label: "HNSW (graph)" },
      { id: "exact", label: "Exact" },
      { id: "metadata", label: "Metadata acceleration" },
      { id: "hybrid", label: "Hybrid fusion" },
      { id: "gpu", label: "GPU family" },
    ]}>
      <p className="lede handwritten-lede">
        Indexes plug in behind <code>IndexPort</code>. Two ship in v1.0 —
        graph (HNSW) and exact — and an optional GPU family lives under{" "}
        <code>src/gpu_engine/</code> behind <code>GpuPort</code>.
      </p>

      <h2 id="metrics">Distance metrics</h2>
      <p>
        Three metrics: <code>cosine</code>, <code>euclidean</code>, and{" "}
        <code>dot</code>. Distance kernels are dispatched through a
        function pointer in <code>Metrics</code> so AVX2 / AVX-512 / NEON
        variants can slot in without touching call sites. By convention,
        smaller distance always means more similar.
      </p>

      <h2 id="hnsw">HNSW (graph)</h2>
      <p>
        <code>HierarchicalGraphIndex</code> implements Hierarchical Navigable
        Small World — a multi-layer proximity graph that gives sub-linear
        ANN with strong recall. Three parameters matter:
      </p>
      <ul>
        <li><code>M</code> / <code>max_connections</code> — outgoing neighbours per node. Higher = better recall, more memory.</li>
        <li><code>ef_construction</code> — candidate pool during insertion. Higher = better graph, slower writes.</li>
        <li><code>ef_search</code> — candidate pool during search. Higher = better recall, slower reads.</li>
      </ul>
      <SketchCard caption="HNSW — sparse hubs at the top, full population at layer 0. Search starts at a hub and descends greedily, refining with ef_search at each step.">
        <HnswLayersDiagram />
      </SketchCard>

      <h2 id="exact">Exact</h2>
      <p>
        <code>ExactIndex</code> performs a brute-force scan. Use it for
        small vaults, ground-truth measurement when tuning recall, and unit
        tests. It implements the same <code>IndexPort</code> contract, so
        switching is a configuration change.
      </p>

      <h2 id="metadata">Metadata acceleration</h2>
      <p>
        <code>MetadataIndex</code> is an exact-match accelerator for
        equality and set-membership predicates. The planner consults it
        first; if it produces a sufficiently small candidate set, the
        executor switches strategy to <code>exact_candidates</code> over
        the narrow set instead of running ANN over the full population.
      </p>
      <CodeBlock lang="python">
{`plan = docs.explain_seek(
    [1.0, 0.0],
    top=5,
    where=elips.Filter().field("kind").equals("design"),
)
print(plan.strategy, plan.metadata_accelerated, plan.candidate_count)`}
      </CodeBlock>

      <h2 id="hybrid">Hybrid fusion</h2>
      <p>
        <code>seek_hybrid()</code> combines vector distance with lexical
        overlap on attached <code>DocumentAttachment</code> text. The
        planner emits the <code>hybrid_fusion</code> strategy; fusion is
        score-level, not retrieval-level, so a record only needs to surface
        in one stage to be considered.
      </p>

      <h2 id="gpu">GPU family</h2>
      <p>
        When built with <code>-DELIPS_GPU_ENABLED=ON</code>, ELIPS exposes
        several GPU index types behind <code>GpuPort</code>:
        <code>GpuBruteForceIndex</code>, <code>GpuGraphIndex</code>,{" "}
        <code>GpuIVFFlatIndex</code>, <code>GpuIVFPQIndex</code>,{" "}
        <code>GpuHybridIndex</code>, and a distributed variant. Selection
        runs through <code>GpuSelector</code> and <code>GpuDeviceManager</code> at
        startup; domain code never sees a backend type. For the full
        engine architecture, see{" "}
        <Link to="/docs/gpu-engine">GPU engine</Link>.
      </p>
    </DocsShell>
  );
}
