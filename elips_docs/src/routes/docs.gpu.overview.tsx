import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/gpu/overview")({
  head: () => ({
    meta: [
      { title: "GPU Engine Overview — ELIPS Hardware Acceleration" },
      {
        name: "description",
        content:
          "Overview of the ELIPS GPU engine: the Metal backend (real), experimental CUDA/HIP, the GpuBackend abstraction, memory pools, query batching, and acceleration pipelines.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="GPU Engine"
      title="GPU Engine Architecture"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "backends", label: "Backend maturity" },
        { id: "port-architecture", label: "GpuBackend abstraction" },
        { id: "index-family", label: "GPU index family" },
        { id: "memory-pools", label: "Memory management & pools" },
        { id: "batching-pipelines", label: "Query batcher & pipelines" },
        { id: "python-integration", label: "Python API integration" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS has a multi-backend GPU abstraction for offloading distance math and top-k selection
        to hardware accelerators. Today the <strong>Metal</strong> backend is the implemented,
        production path on Apple silicon; CUDA and HIP/ROCm are experimental, and everything falls
        back to CPU.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        The GPU engine accelerates vector search by running distance and top-k kernels on the
        device. On Apple silicon, unified memory lets ELIPS skip host↔device copies. The exact
        brute-force path is reachable directly from Python through{" "}
        <code>elips.bf_knn(..., device="metal")</code>.
      </p>

      <h2 id="backends">Backend maturity</h2>
      <div className="not-prose my-4 overflow-x-auto">
        <table className="w-full text-[13.5px] border border-hairline">
          <thead>
            <tr className="bg-surface text-ink">
              <th className="text-left p-2 border-b border-hairline">Backend</th>
              <th className="text-left p-2 border-b border-hairline">Status</th>
              <th className="text-left p-2 border-b border-hairline">Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="p-2 border-b border-hairline">Metal</td>
              <td className="p-2 border-b border-hairline">Production</td>
              <td className="p-2 border-b border-hairline">
                Apple M-series, default ON. Distance + top-k kernels, unified memory.
              </td>
            </tr>
            <tr>
              <td className="p-2 border-b border-hairline">CUDA</td>
              <td className="p-2 border-b border-hairline">Experimental</td>
              <td className="p-2 border-b border-hairline">
                NVIDIA. Off by default; opt in and expect rough edges.
              </td>
            </tr>
            <tr>
              <td className="p-2 border-b border-hairline">HIP / ROCm</td>
              <td className="p-2 border-b border-hairline">Experimental</td>
              <td className="p-2 border-b border-hairline">AMD. Off by default.</td>
            </tr>
            <tr>
              <td className="p-2">CPU</td>
              <td className="p-2">Always</td>
              <td className="p-2">The fallback path; present in every build.</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        SYCL and Vulkan backends were removed from the engine; they are not part of ELIPS.
      </p>

      <h2 id="port-architecture">GpuBackend abstraction</h2>
      <CodeBlock lang="cpp">{`namespace elips::gpu {
    class GpuBackend {
    public:
        virtual ~GpuBackend() = default;
        virtual GpuDeviceInfo device_info() const = 0;
        virtual GpuMetricsSnapshot metrics() const = 0;
        virtual void compute_distances(
            const float* queries, std::size_t num_queries,
            const float* database, std::size_t num_vectors,
            std::size_t dim, float* distances
        ) = 0;
    };
}`}</CodeBlock>
      <p>
        A <code>BackendSelector</code> picks the backend once at startup; the chain is backend →
        device → CPU, so the runtime always lands on a working path.
      </p>

      <h2 id="index-family">GPU Index Family</h2>
      <p>
        ELIPS implements dedicated GPU index ports under <code>include/elips/gpu_engine/</code>:
      </p>
      <ul>
        <li>
          <code>GpuBruteForceIndex</code> — High-throughput exact k-NN distance computation.
        </li>
        <li>
          <code>GpuIVFFlatIndex</code> — Inverted File index with GPU cluster lookup.
        </li>
        <li>
          <code>GpuIVFPQIndex</code> — Product Quantization with GPU centroid lookup table decoding.
        </li>
        <li>
          <code>GpuGraphIndex</code> — GPU-assisted HNSW graph entry point evaluation.
        </li>
        <li>
          <code>GpuHybridIndex</code> — Concurrent GPU vector similarity and sparse text relevance ranking.
        </li>
      </ul>

      <h2 id="memory-pools">Memory Management & Pools</h2>
      <p>
        To avoid high latency associated with device memory allocation (`cudaMalloc`), ELIPS uses custom allocators:
      </p>
      <ul>
        <li>
          <code>GpuMemoryPool</code> — Pre-allocates slab chunks on device memory.
        </li>
        <li>
          <code>PinnedBuffer</code> — Host page-locked memory for zero-copy DMA transfers.
        </li>
        <li>
          <code>UnifiedBuffer</code> — Unified Virtual Memory (UVM) bridging CPU and GPU pointers.
        </li>
      </ul>

      <h2 id="batching-pipelines">Query batcher &amp; pipelines</h2>
      <CodeBlock lang="cpp">{`namespace elips::gpu {
    class QueryBatcher;
    class GpuIngestionPipeline;
    class GpuSearchPipeline;
}`}</CodeBlock>
      <p>
        The <code>QueryBatcher</code> coalesces incoming single-vector queries from multiple CPU
        worker threads into contiguous GPU batch matrices, amortising kernel-launch overhead.
      </p>

      <h2 id="python-integration">Python API integration</h2>
      <CodeBlock lang="python">{`import elips

# Select a backend by device string (not gpu=True)
with elips.PersistentClient("./db", dimension=1536, device="metal") as client:
    info = client.native.gpu_info()
    print(info.backend)   # "metal" on Apple silicon, else "cpu"

# Enumerate accelerators without opening a database
for spec in elips.accelerators():
    print(spec.name, spec.backend, spec.unified_memory)`}</CodeBlock>
    </DocsShell>
  );
}
