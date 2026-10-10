import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";
import { SketchCard, GpuEngineDiagram, DynamicBatcherDiagram } from "../components/SketchDiagram";

export const Route = createFileRoute("/docs/gpu-engine")({
  head: () => ({
    meta: [
      { title: "GPU engine — ELIPS Docs" },
      {
        name: "description",
        content:
          "The ELIPS GPU engine — the GpuBackend port, backends (Metal production on Apple, CUDA/HIP experimental), distance + top-k kernels, bf_knn, query batching, memory pools, and configuration.",
      },
      { property: "og:title", content: "GPU engine — ELIPS" },
      {
        property: "og:description",
        content:
          "Optional GPU acceleration behind GpuBackend — Metal (production on Apple Silicon), CUDA and HIP/ROCm (experimental) — running distance and top-k kernels with query batching and pinned pools.",
      },
      { property: "og:url", content: "/docs/gpu-engine" },
    ],
    links: [{ rel: "canonical", href: "/docs/gpu-engine" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Internals"
      title="GPU engine"
      toc={[
        { id: "thesis", label: "Thesis" },
        { id: "architecture", label: "Layered architecture" },
        { id: "ports", label: "Ports (ISP)" },
        { id: "config", label: "GpuConfig" },
        { id: "backends", label: "Backends & maturity" },
        { id: "indexes", label: "Distance & top-k path" },
        { id: "batcher", label: "Query batcher" },
        { id: "memory", label: "Memory" },
        { id: "errors", label: "Error model" },
        { id: "metrics", label: "Metrics & profiling" },
        { id: "build", label: "Build flags" },
      ]}
    >
      <p className="lede handwritten-lede">
        ELIPS treats the GPU as an optional engine — never a runtime dependency. Domain code talks
        to <code>Index</code>; when GPU is built in, the distance and top-k stage{" "}
        <em>happens</em> to run on the device. Everything else — selection, allocation, batching,
        fallback — is out of band, behind narrow ports.
      </p>

      <h2 id="thesis">Thesis</h2>
      <p>
        Three commitments shape every header under <code>include/elips/gpu_engine/</code>:
      </p>
      <ul>
        <li>
          <strong>Interface Segregation.</strong> Four ports (compute, memory, kernel, stream)
          instead of one giant facade.
        </li>
        <li>
          <strong>Failure is expected.</strong> Every GPU operation returns{" "}
          <code>std::expected&lt;T, GpuError&gt;</code> — device-lost, OOM, and unsupported metric
          are values, not exceptions.
        </li>
        <li>
          <strong>Optional, additive.</strong> The CPU index and serving path stand on their own.
          GPU is a build-time switch (<code>-DELIPS_GPU_ENABLED=ON</code>), not a runtime
          requirement.
        </li>
      </ul>

      <h2 id="architecture">Layered architecture</h2>
      <SketchCard caption="Surface → orchestration → ports → backends. Each layer only knows the layer directly beneath it.">
        <GpuEngineDiagram />
      </SketchCard>

      <h2 id="ports">Ports — interface segregation</h2>
      <p>
        The engine is decomposed across five virtual interfaces so each caller depends only on the
        slice it actually needs.
      </p>
      <CodeBlock lang="cpp">
        {`// GpuBackend.hpp — the umbrella port (init + compute + top_k)
class GpuBackend {
public:
    virtual std::expected<void, GpuError>      initialize(const GpuConfig&) = 0;
    virtual void                               shutdown() noexcept = 0;
    virtual GpuDeviceInfo                      device_info() const noexcept = 0;

    virtual std::expected<GpuBuffer, GpuError> allocate_device(size_t bytes) = 0;
    virtual void                               free_device(GpuBuffer) noexcept = 0;
    virtual std::expected<void, GpuError>      upload(const void*, GpuBuffer&, size_t) = 0;
    virtual std::expected<void, GpuError>      download(const GpuBuffer&, void*, size_t) = 0;

    virtual std::expected<void, GpuError>      compute_distances_batch(
        std::span<const float> queries, std::span<const float> database,
        std::span<float> out, size_t nq, size_t nb, size_t dim, elips::Metric) = 0;

    virtual std::expected<void, GpuError>      top_k(
        std::span<const float> distances,
        std::span<uint32_t> indices_out, std::span<float> values_out,
        size_t nq, size_t nb, size_t k) = 0;

    virtual void synchronize() = 0;
    virtual bool is_idle() const noexcept = 0;
};`}
      </CodeBlock>
      <p>The dedicated ports keep call sites honest:</p>
      <ul>
        <li>
          <code>GpuMemoryPort</code> — <code>allocate</code> / <code>deallocate</code> / pinned host
          buffers / <code>bytes_used / available / peak</code>.
        </li>
        <li>
          <code>GpuKernelPort</code> — explicit metric-typed kernels: <code>cosine_fp32</code>,{" "}
          <code>euclidean_fp32</code>, <code>dot_product_fp32</code>.
        </li>
        <li>
          <code>GpuStreamPort</code> — fine-grained synchronisation surface for callers that
          schedule their own streams.
        </li>
      </ul>
      <p>
        There is no GPU-resident ANN index. The device accelerates the exact distance + top-k stage;
        the exact brute-force path is surfaced to Python as{" "}
        <code>elips.bf_knn(xq, xb, k, device=&quot;metal&quot;)</code>. HNSW and IVF indexes stay on
        the CPU.
      </p>

      <h2 id="config">GpuConfig</h2>
      <CodeBlock lang="cpp">
        {`enum class GpuPolicy    { Auto, PreferGpu, RequireGpu, CpuOnly, Specific };
enum class GpuPrecision { FP32, FP16, Auto };

struct GpuConfig {
    GpuPolicy    policy{GpuPolicy::Auto};
    std::string  preferred_backend;        // "metal" | "cuda" | "hip" | ""
    int32_t      device_index{-1};         // -1 = let the selector decide

    size_t       device_memory_pool_bytes{0};            // 0 = auto-size
    size_t       pinned_host_pool_bytes{256UL * 1024 * 1024};
    bool         use_unified_memory{false};              // Apple Silicon

    size_t       query_batch_window_us{500};
    size_t       query_batch_max_size{256};
    GpuPrecision search_precision{GpuPrecision::Auto};
    bool         enable_fp16_search{false};

    bool         enable_profiling{false};
    bool         emit_kernel_timings{false};
};`}
      </CodeBlock>
      <p>
        <code>policy</code> drives the selector: <code>Auto</code> tries GPU and falls back;{" "}
        <code>PreferGpu</code> warns but still falls back; <code>RequireGpu</code> errors if no
        device is usable; <code>CpuOnly</code> short-circuits the engine entirely;{" "}
        <code>Specific</code> pins to <code>preferred_backend</code> / <code>device_index</code>.
        From Python this is reached with <code>Client(..., device=&quot;metal&quot;, fallback=False)</code>{" "}
        or <code>gpu=GpuConfig(...)</code> — never <code>gpu=True</code>.
      </p>

      <h2 id="backends">Backends &amp; maturity</h2>
      <p>
        Three concrete backends live under <code>src/gpu_engine/backends/</code>:
      </p>
      <ul>
        <li>
          <strong>Metal</strong> (<code>backends/metal/</code>) — Apple Silicon, the production path.
          Leans on unified memory (<code>has_unified_memory</code> in <code>GpuDeviceInfo</code>).
        </li>
        <li>
          <strong>CUDA</strong> (<code>backends/cuda/</code>) — NVIDIA path, ships{" "}
          <code>cosine_fp32.cu</code> kernel. Experimental.
        </li>
        <li>
          <strong>HIP / ROCm</strong> (<code>backends/hip/</code>) — AMD path, parallel{" "}
          <code>cosine_fp32.hip</code> kernel. Experimental.
        </li>
      </ul>
      <table>
        <thead>
          <tr>
            <th>Backend</th>
            <th>Status</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Metal</td>
            <td>Production</td>
            <td>Default ON on Apple; distance + top-k kernels.</td>
          </tr>
          <tr>
            <td>CUDA</td>
            <td>Experimental</td>
            <td>NVIDIA; opt-in.</td>
          </tr>
          <tr>
            <td>HIP / ROCm</td>
            <td>Experimental</td>
            <td>AMD; opt-in.</td>
          </tr>
          <tr>
            <td>CPU</td>
            <td>Always</td>
            <td>Fallback and default when GPU is not built in.</td>
          </tr>
        </tbody>
      </table>
      <p>
        Each backend implements <code>GpuBackend</code>, registers its capabilities in{" "}
        <code>GpuDeviceInfo</code> (FP16 support, unified memory, batching) and is ranked by{" "}
        <code>BackendSelector</code> against the accelerators ELIPS enumerates. From Python, that
        list is <code>elips.accelerators()</code>:
      </p>
      <CodeBlock lang="python">
        {`import elips

for acc in elips.accelerators():          # list[AcceleratorSpec]
    print(acc.name, acc.backend, acc.unified_memory, acc.supports_fp16)

print(elips.has_gpu)                       # bool
best = elips.accelerator()                 # AcceleratorSpec | None

# pin a device; raise BackendUnavailableError if absent and fallback is off
client = elips.PersistentClient("/tmp/db", dimension=384, device="metal", fallback=False)
print(client.native.gpu_info())`}
      </CodeBlock>

      <h2 id="indexes">Distance &amp; top-k path</h2>
      <p>
        The GPU does one job: the exact distance computation and the top-k selection over a batch of
        queries. There is no GPU-resident ANN structure — no CAGRA, no device-side IVF — so the
        kernels stay small and portable across Metal, CUDA, and HIP. The exact brute-force path is
        exposed to Python as <code>bf_knn</code>, and ANN continues to run on the CPU indexes
        (<code>IndexHNSW</code>, <code>IndexIVFFlat</code>, …).
      </p>
      <CodeBlock lang="python">
        {`import numpy as np, elips

xb = np.random.rand(100000, 128).astype("float32")
xq = np.random.rand(1000, 128).astype("float32")

# exact k-NN on the Apple GPU, no index build
D, I = elips.bf_knn(xq, xb, 10, metric="l2", device="metal")`}
      </CodeBlock>
      <p>
        On the shipped Apple M4 benchmark (dim 128, nb 100000, nq 1000, k 10, L2, ELIPS 1.1.0),{" "}
        <code>bf_knn</code> on Metal reaches recall 1.000 at 914 QPS batched — versus 24 QPS when
        queries are issued one at a time, which is exactly what the query batcher below is for.
      </p>

      <h2 id="batcher">Query batcher</h2>
      <p>
        Concurrent single-query searches coalesce into one kernel launch.{" "}
        <code>QueryBatcher</code> opens a window of <code>query_batch_window_us</code> µs (default
        500), buffers up to <code>query_batch_max_size</code> queries (default 256), then flushes —
        either when the window closes or the buffer fills.
      </p>
      <SketchCard caption="A short batching window converts many small queries into a few large kernel launches. Each query still resolves through its own std::future.">
        <DynamicBatcherDiagram />
      </SketchCard>
      <CodeBlock lang="cpp">
        {`std::future<std::vector<SearchResult>>
QueryBatcher::enqueue(std::span<const float> q, size_t k);

struct BatchStats {
    size_t queries_coalesced{0};
    size_t kernel_launches{0};
    float  avg_batch_size{0.0f};
    float  p99_latency_us{0.0f};
};`}
      </CodeBlock>

      <h2 id="memory">Memory</h2>
      <p>
        <code>GpuMemoryManager</code> implements <code>GpuMemoryPort</code>: a slab-allocator over a
        device-side pool sized by <code>device_memory_pool_bytes</code>, plus a pinned host pool
        sized by <code>pinned_host_pool_bytes</code> for fast H2D / D2H transfers.{" "}
        <code>GpuMemoryPool</code> provides a simpler in-pool allocator used by the search pipeline
        for scratch buffers.
      </p>
      <p>
        On Apple Silicon, <code>use_unified_memory</code> avoids redundant copies — the same
        allocation is visible to host and GPU. See ADR-GPU-006.
      </p>

      <h2 id="errors">Error model</h2>
      <CodeBlock lang="cpp">
        {`enum class GpuError {
    DeviceNotFound, InsufficientMemory, KernelLaunchFailed,
    TransferFailed, IndexBuildFailed, UnsupportedMetric,
    InitializationFailed, BackendUnavailable,
};

[[nodiscard]] std::expected<...> op(...);   // every port method`}
      </CodeBlock>
      <p>
        Errors flow up as values; the caller decides whether to fall back to CPU, retry on another
        device, or surface. The fallback chain is codified in ADR-GPU-008.
      </p>

      <h2 id="metrics">Metrics &amp; profiling</h2>
      <p>
        <code>client.native.gpu_info()</code> (on the native <code>Database</code> handle) returns a{" "}
        <code>GpuMetricsSnapshot</code>:
      </p>
      <CodeBlock lang="cpp">
        {`struct GpuMetricsSnapshot {
    std::string backend, device_name;
    size_t device_memory_used_bytes, device_memory_total_bytes;
    size_t search_kernel_launches_total;
    size_t search_p50_latency_us, search_p99_latency_us;
    float  batch_avg_size, batch_coalescing_ratio;
    bool   fp16_search_enabled;
    size_t fallback_events_total;
    size_t kernel_errors_total;
    size_t pinned_memory_pool_used_bytes;
};`}
      </CodeBlock>
      <p>
        Per-kernel timing is opt-in via <code>GpuConfig::emit_kernel_timings</code>; recent samples
        are retrievable from <code>GpuProfiler::recent_timings(n)</code>.
      </p>

      <h2 id="build">Build flags</h2>
      <CodeBlock lang="bash">
        {`# CPU-only — the default
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# GPU-enabled — backends auto-detected from the host toolchain
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DELIPS_GPU_ENABLED=ON

cmake --build build -j
ctest --test-dir build --output-on-failure`}
      </CodeBlock>
      <p>
        Reference: ADR-GPU-001 through ADR-GPU-008 in{" "}
        <Link to="/docs/design-decisions">Design decisions</Link>.
      </p>
    </DocsShell>
  );
}
