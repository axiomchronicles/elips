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
          "Comprehensive overview of ELIPS GPU engine: CUDA, HIP, Metal backends, GpuPort architecture, memory pools, dynamic batching, and acceleration pipelines.",
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
        { id: "backends", label: "Backend Support (CUDA, HIP, Metal)" },
        { id: "port-architecture", label: "GpuPort & Device Abstractions" },
        { id: "index-family", label: "GPU Index Family" },
        { id: "memory-pools", label: "Memory Management & Pools" },
        { id: "batching-pipelines", label: "Dynamic Batcher & Pipelines" },
        { id: "python-integration", label: "Python API Integration" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS features a unified, multi-backend GPU acceleration engine capable of offloading matrix distance calculations, IVF-PQ quantization, graph traversals, and dynamic batch queries to hardware accelerators.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        The GPU engine provides ultra-high throughput vector search by exploiting massively parallel CUDA cores (NVIDIA), HIP streams (AMD), and Metal Performance Shaders (Apple Silicon).
      </p>

      <h2 id="backends">Backend Support (CUDA, HIP, Metal)</h2>
      <ul>
        <li>
          <strong>CUDA Backend:</strong> Native CUDA C++ kernels targeting NVIDIA Volta, Ampere, Hopper, and Blackwell architectures using async streams and Tensor Cores.
        </li>
        <li>
          <strong>HIP Backend:</strong> AMD ROCm acceleration for Instinct MI200/MI300 series accelerators.
        </li>
        <li>
          <strong>Metal Backend:</strong> Hardware-accelerated matrix multiplication on Apple M1/M2/M3/M4 unified memory architectures.
        </li>
      </ul>

      <h2 id="port-architecture">GpuPort & Device Abstractions</h2>
      <CodeBlock lang="cpp">{`namespace elips::gpu {
    class GpuPort {
    public:
        virtual ~GpuPort() = default;
        virtual GpuDeviceInfo device_info() const = 0;
        virtual GpuMetricsSnapshot metrics() const = 0;
        virtual void compute_distances(
            const float* queries, std::size_t num_queries,
            const float* database, std::size_t num_vectors,
            std::size_t dim, float* distances
        ) = 0;
    };
}`}</CodeBlock>

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

      <h2 id="batching-pipelines">Dynamic Batcher & Pipelines</h2>
      <CodeBlock lang="cpp">{`namespace elips::gpu {
    class DynamicBatcher;
    class GpuIngestionPipeline;
    class GpuSearchPipeline;
}`}</CodeBlock>
      <p>
        The <code>DynamicBatcher</code> accumulates incoming single vector queries from multiple CPU worker threads into contiguous GPU batch matrices, maximizing Tensor Core compute efficiency.
      </p>

      <h2 id="python-integration">Python API Integration</h2>
      <CodeBlock lang="python">{`import elips

# Initialize GPU accelerator handle
with elips.connect(":memory:", dimension=1536, gpu=True) as engine:
    print(engine.gpu_info())
    # {'device_name': 'NVIDIA RTX 4090', 'total_memory_mb': 24576, 'backend': 'cuda'}`}</CodeBlock>
    </DocsShell>
  );
}
