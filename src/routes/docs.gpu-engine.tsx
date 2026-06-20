import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";
import { SketchCard, GpuEngineDiagram, DynamicBatcherDiagram } from "../components/SketchDiagram";

export const Route = createFileRoute("/docs/gpu-engine")({
  head: () => ({
    meta: [
      { title: "GPU engine — ELIPS Docs" },
      { name: "description", content: "The ELIPS GPU engine — ports, backends (CUDA/HIP/Metal), index family, dynamic batching, memory pools, profiling, and configuration." },
      { property: "og:title", content: "GPU engine — ELIPS" },
      { property: "og:description", content: "Layered GPU acceleration behind GpuPort — CUDA, HIP/ROCm, Metal — with dynamic batching, pinned pools, and brute-force / IVF / graph / hybrid / distributed indexes." },
      { property: "og:url", content: "/docs/gpu-engine" },
    ],
    links: [{ rel: "canonical", href: "/docs/gpu-engine" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell eyebrow="Internals" title="GPU engine" toc={[
      { id: "thesis", label: "Thesis" },
      { id: "architecture", label: "Layered architecture" },
      { id: "ports", label: "Ports (ISP)" },
      { id: "config", label: "GpuConfig" },
      { id: "backends", label: "Backends" },
      { id: "indexes", label: "Index family" },
      { id: "batcher", label: "Dynamic batcher" },
      { id: "memory", label: "Memory" },
      { id: "transfer", label: "Index transfer" },
      { id: "errors", label: "Error model" },
      { id: "metrics", label: "Metrics & profiling" },
      { id: "build", label: "Build flags" },
    ]}>
      <p className="lede handwritten-lede">
        ELIPS treats the GPU as an optional engine — never a runtime
        dependency. Domain code talks to <code>IndexPort</code>; when GPU
        is built in, the index <em>happens</em> to live in device memory.
        Everything else — selection, allocation, batching, fallback — is
        out of band, behind narrow ports.
      </p>

      <h2 id="thesis">Thesis</h2>
      <p>
        Three commitments shape every header under{" "}
        <code>include/elips/gpu_engine/</code>:
      </p>
      <ul>
        <li><strong>Interface Segregation.</strong> Five ports (compute, memory, kernel, stream, index) instead of one giant facade.</li>
        <li><strong>Failure is expected.</strong> Every GPU operation returns <code>std::expected&lt;T, GpuError&gt;</code> — device-lost, OOM, and unsupported metric are values, not exceptions.</li>
        <li><strong>Optional, additive.</strong> The CPU index and serving path stand on their own. GPU is a build-time switch (<code>-DELIPS_GPU_ENABLED=ON</code>), not a runtime requirement.</li>
      </ul>

      <h2 id="architecture">Layered architecture</h2>
      <SketchCard caption="Surface → orchestration → ports → backends. Each layer only knows the layer directly beneath it.">
        <GpuEngineDiagram />
      </SketchCard>

      <h2 id="ports">Ports — interface segregation</h2>
      <p>
        The engine is decomposed across five virtual interfaces so each
        caller depends only on the slice it actually needs.
      </p>
      <CodeBlock lang="cpp">
{`// GpuPort.hpp — the umbrella port (init + compute + top_k)
class GpuPort {
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
        <li><code>GpuMemoryPort</code> — <code>allocate</code> / <code>deallocate</code> / pinned host buffers / <code>bytes_used / available / peak</code>.</li>
        <li><code>GpuKernelPort</code> — explicit metric-typed kernels: <code>cosine_fp32</code>, <code>euclidean_fp32</code>, <code>dot_product_fp32</code>.</li>
        <li><code>GpuStreamPort</code> — fine-grained synchronisation surface for callers that schedule their own streams.</li>
        <li><code>GpuIndexPort</code> — implements both <code>IndexPort</code> and <code>IndexTransferPort</code>; exposes <code>build_from_batch</code>, <code>search_batch</code>, <code>export_to_cpu_index</code> / <code>import_from_cpu_index</code>, plus <code>backend_name()</code> and <code>device_bytes_used()</code>.</li>
      </ul>

      <h2 id="config">GpuConfig</h2>
      <CodeBlock lang="cpp">
{`enum class GpuPolicy        { Auto, PreferGpu, RequireGpu, CpuOnly, Specific };
enum class IndexBuildMode   { GpuBuild_CpuServe, GpuBuild_GpuServe, Hybrid };
enum class GpuIndexAlgorithm{ Auto, CagraGraph, IvfFlat, IvfPq, BruteForce };
enum class GpuPrecision     { FP32, FP16, Int8, Auto };

struct GraphIndexBuildParams {
    size_t intermediate_graph_degree{128};
    size_t graph_degree{64};
    enum class BuildAlgo { IvfPq, NnDescent, IterativeSearch } build_algo{...};
    size_t nn_descent_iterations{20};
    float  compression_ratio{0.0f};
};
struct IvfPqBuildParams {
    size_t   n_lists{1024};
    uint32_t pq_dim{0};
    uint32_t pq_bits{8};
    bool     add_data_on_build{true};
    size_t   kmeans_n_iters{20};
    float    kmeans_trainset_fraction{0.5f};
};

struct GpuConfig {
    GpuPolicy           policy{GpuPolicy::Auto};
    std::string         preferred_backend;        // "cuda" | "hip" | "metal" | ""
    int32_t             device_index{-1};         // -1 = let selector decide
    IndexBuildMode      index_build_mode{IndexBuildMode::GpuBuild_GpuServe};
    GpuIndexAlgorithm   algorithm{GpuIndexAlgorithm::Auto};

    size_t              device_memory_pool_bytes{0};            // 0 = auto-size
    size_t              pinned_host_pool_bytes{256UL * 1024 * 1024};
    bool                use_unified_memory{false};               // Apple / Grace Hopper

    size_t              default_ef_search_gpu{64};
    size_t              dynamic_batch_window_us{500};
    size_t              dynamic_batch_max_size{256};
    bool                enable_fp16_search{false};

    GraphIndexBuildParams graph_params;
    IvfPqBuildParams      ivf_pq_params;
    GpuPrecision          search_precision{GpuPrecision::Auto};

    bool  auto_rebuild_on_startup{false};
    float rebuild_threshold_ratio{0.1f};

    bool  enable_profiling{false};
    bool  emit_kernel_timings{false};
};`}
      </CodeBlock>
      <p>
        <code>policy</code> drives the selector:{" "}
        <code>Auto</code> tries GPU and falls back; <code>PreferGpu</code> warns
        but still falls back; <code>RequireGpu</code> errors if no device is
        usable; <code>CpuOnly</code> short-circuits the engine entirely;{" "}
        <code>Specific</code> pins to <code>preferred_backend</code> /{" "}
        <code>device_index</code>.
      </p>

      <h2 id="backends">Backends</h2>
      <p>
        Three concrete backends live under{" "}
        <code>src/gpu_engine/backends/</code>:
      </p>
      <ul>
        <li><strong>CUDA</strong> (<code>backends/cuda/</code>) — NVIDIA path, ships <code>cosine_fp32.cu</code> kernel.</li>
        <li><strong>HIP / ROCm</strong> (<code>backends/hip/</code>) — AMD path, parallel <code>cosine_fp32.hip</code> kernel.</li>
        <li><strong>Metal</strong> (<code>backends/metal/</code>) — Apple Silicon, leans on unified memory (<code>has_unified_memory</code> in <code>GpuDeviceInfo</code>).</li>
      </ul>
      <p>
        Each backend implements <code>GpuPort</code>, registers its capabilities
        in <code>GpuDeviceInfo</code> (FP16/BF16/Int8 support, CAGRA, IVF-PQ,
        dynamic batching, half-precision search) and is selected by
        <code>GpuSelector::rank_backend</code> together with the device list
        returned by <code>GpuDeviceManager</code>.
      </p>

      <h2 id="indexes">Index family</h2>
      <p>
        Every GPU index is a final subclass of <code>GpuIndexPort</code>, which
        itself extends both <code>IndexPort</code> and <code>IndexTransferPort</code>.
        Domain code keeps using <code>IndexPort</code>.
      </p>
      <ul>
        <li><code>GpuBruteForceIndex</code> — exhaustive search on device, ideal for small/medium vaults and ground-truth probes.</li>
        <li><code>GpuIVFFlatIndex</code> — inverted-file with flat lists; tunable <code>n_lists</code>.</li>
        <li><code>GpuIVFPQIndex</code> — inverted-file with product quantisation (<code>pq_dim</code>, <code>pq_bits</code>); large-scale, memory-efficient.</li>
        <li><code>GpuGraphIndex</code> — CAGRA-style graph; build via IVF-PQ, NN-descent, or iterative search.</li>
        <li><code>GpuHybridIndex</code> — composite for mixed dense / lexical paths.</li>
        <li><code>GpuDistributedIndex</code> — multi-GPU index with a <code>DistributedMode</code> selector.</li>
      </ul>

      <h2 id="batcher">Dynamic batcher</h2>
      <p>
        Concurrent single-query searches coalesce into one kernel launch.{" "}
        <code>DynamicBatcher</code> opens a window of{" "}
        <code>dynamic_batch_window_us</code> µs (default 500), buffers up to{" "}
        <code>dynamic_batch_max_size</code> queries (default 256), then flushes —
        either when the window closes or the buffer fills.
      </p>
      <SketchCard caption="A short batching window converts many small queries into a few large kernel launches. Each query still resolves through its own std::future.">
        <DynamicBatcherDiagram />
      </SketchCard>
      <CodeBlock lang="cpp">
{`std::future<std::vector<SearchResult>>
DynamicBatcher::enqueue(std::span<const float> q, size_t k);

struct BatchStats {
    size_t queries_coalesced{0};
    size_t kernel_launches{0};
    float  avg_batch_size{0.0f};
    float  p99_latency_us{0.0f};
};`}
      </CodeBlock>

      <h2 id="memory">Memory</h2>
      <p>
        <code>GpuMemoryManager</code> implements <code>GpuMemoryPort</code>: a
        slab-allocator over a device-side pool sized by{" "}
        <code>device_memory_pool_bytes</code>, plus a pinned host pool
        sized by <code>pinned_host_pool_bytes</code> for fast H2D / D2H
        transfers. <code>GpuMemoryPool</code> provides a simpler in-pool
        allocator used by the search pipeline for scratch buffers.
      </p>
      <p>
        On Apple Silicon, <code>use_unified_memory</code> avoids redundant
        copies — the same allocation is visible to host and GPU. See
        ADR-GPU-006.
      </p>

      <h2 id="transfer">Index transfer</h2>
      <p>
        <code>GpuIndexTransferManager</code> moves built indexes between
        device and host:
      </p>
      <CodeBlock lang="cpp">
{`clone(source, destination);                 // CPU↔CPU or GPU↔GPU
clone_cpu_to_gpu(cpu_src, gpu_dst);         // GpuBuild_CpuServe → upload
clone_gpu_to_cpu(gpu_src, cpu_dst);         // export for serving on CPU`}
      </CodeBlock>
      <p>
        Together with <code>IndexBuildMode::GpuBuild_CpuServe</code>, this
        unlocks the canonical pattern: build a CAGRA graph on a beefy
        training box, then serve it from CPU nodes with no GPU dependency.
        See <Link to="/docs/design-decisions">ADR-GPU-003</Link>.
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
        Errors flow up as values; the caller decides whether to fall back
        to CPU, retry on another device, or surface. The fallback chain is
        codified in ADR-GPU-008.
      </p>

      <h2 id="metrics">Metrics &amp; profiling</h2>
      <p>
        <code>gpu_stats()</code> on <code>ElipsInstance</code> returns a{" "}
        <code>GpuMetricsSnapshot</code>:
      </p>
      <CodeBlock lang="cpp">
{`struct GpuMetricsSnapshot {
    std::string backend, device_name;
    size_t device_memory_used_bytes, device_memory_total_bytes;
    size_t index_build_count, index_build_time_total_ms;
    float  index_build_speedup_vs_cpu_avg;
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
        Per-kernel timing is opt-in via{" "}
        <code>GpuConfig::emit_kernel_timings</code>; recent samples are
        retrievable from <code>GpuProfiler::recent_timings(n)</code>.
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
        Reference: ADR-GPU-001 through ADR-GPU-010 in{" "}
        <Link to="/docs/design-decisions">Design decisions</Link>.
      </p>
    </DocsShell>
  );
}
