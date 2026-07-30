import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/gpu")({
  head: () => ({
    meta: [
      { title: "GPU API - ELIPS Python Docs" },
      { name: "description", content: "Documentation for the ELIPS Python GPU API." },
    ],
    links: [],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="GPU API"
      toc={[
        { id: "install", label: "Installation & Builds" },
        { id: "two-surfaces", label: "Core vs Modern API" },
        { id: "device-discovery", label: "Device Discovery" },
        { id: "capacity-planning", label: "Capacity Planning" },
        { id: "gpu-select", label: "Selecting a GPU" },
        { id: "gpu-device", label: "Low-level Device API" },
        { id: "gpu-config", label: "Configuration & Enums" },
        { id: "modern-accelerator", label: "Modern Accelerator API" },
        { id: "brute-force-example", label: "End-to-End Example" },
        { id: "batch-stats", label: "Batch Statistics" },
        { id: "database-integration", label: "Database Integration" },
        { id: "errors", label: "Error Handling" },
        { id: "pitfalls", label: "Common Pitfalls" },
      ]}
    >
      <p>
        ELIPS provides powerful GPU acceleration for high-performance similarity search and indexing. See the <Link to="/docs/gpu-engine">C++ GPU Engine internals</Link> and the <Link to="/docs/python-sdk">Python SDK Base API</Link> for more context.
      </p>

      <h2 id="install">Installation & Builds</h2>
      <p>
        GPU support requires specific hardware and drivers:
      </p>
      <ul>
        <li><strong>Metal:</strong> Apple Silicon only. Gated by <code>-DELIPS_GPU_METAL=ON</code> (defaults to ON for Apple platforms). Do not force this on Linux.</li>
        <li><strong>CUDA:</strong> Requires NVIDIA Toolkit.</li>
        <li><strong>HIP:</strong> Requires ROCm.</li>
      </ul>

      <h2 id="two-surfaces">Two Surfaces: Core vs Modern API</h2>
      <p>
        The GPU functionality is exposed via two layers:
      </p>
      <ul>
        <li><strong>Low-level (<code>elips</code> core module):</strong> Direct bindings to the C++ GPU engine, suitable for fine-grained control and zero-overhead interop.</li>
        <li><strong>Modern wrapper (<code>elips._modern.gpu</code>):</strong> A pythonic abstraction via dataclasses and simplified methods for the majority of use cases.</li>
      </ul>

      <h2 id="device-discovery">Device Discovery</h2>
      <p>
        Discover available hardware across all compiled backends.
      </p>
      <h3>Modern API</h3>
      <ul>
        <li><code>accelerators() -&gt; list[AcceleratorSpec]</code>: List all GPUs detected on the system.</li>
      </ul>
      <h3>Core API</h3>
      <ul>
        <li><code>gpu_devices() -&gt; list[GpuDeviceInfo]</code>: Probe all GPU backends.</li>
        <li><code>gpu_cpu_fallback_info() -&gt; GpuDeviceInfo</code>: Details for CPU fallback when no GPU is available or selected.</li>
        <li><code>gpu_runtime_device_info() -&gt; GpuDeviceInfo</code>: The device this process would pick by default.</li>
      </ul>

      <h2 id="capacity-planning">Capacity Planning</h2>
      <p>
        Determine if a dataset will fit into VRAM.
      </p>
      <h3>Modern API</h3>
      <ul>
        <li><code>AcceleratorSpec.can_fit(n_vectors, dimension, config=None) -&gt; bool</code>: Predicts if the specified vectors fit in the device memory.</li>
      </ul>
      <h3>Core API</h3>
      <ul>
        <li><code>gpu_can_fit_index(device, n_vectors, dimension, config=...) -&gt; bool</code>: Static capacity check for a given device info object.</li>
      </ul>

      <h2 id="gpu-select">Selecting a GPU</h2>
      <p>
        Initialize a GPU backend for compute.
      </p>
      <h3>Modern API</h3>
      <ul>
        <li><code>accelerator(config=None) -&gt; Optional[Accelerator]</code>: Select and initialize the best available GPU. Returns <code>None</code> if no GPU is available.</li>
      </ul>
      <h3>Core API</h3>
      <ul>
        <li><code>gpu_select(config=...) -&gt; Optional[GpuDevice]</code>: Initialize the core device handle. Always check for <code>None</code>.</li>
      </ul>

      <h2 id="gpu-device">Low-level Device API (<code>GpuDevice</code>)</h2>
      <p>
        The <code>GpuDevice</code> class represents an active device handle.
      </p>
      <ul>
        <li><strong>Properties:</strong> <code>device_info</code>, <code>available</code>, <code>idle</code>, <code>backend</code>, <code>memory (GpuMemory)</code>, <code>profiler (GpuProfiler)</code>, <code>closed</code>.</li>
        <li><strong>Methods:</strong> 
          <ul>
            <li><code>synchronize()</code>: Block host until device is idle.</li>
            <li><code>compute_distances(queries, database, metric)</code>: Raw distance computation.</li>
            <li><code>top_k(distances, k)</code>: Select top-K from raw distances.</li>
            <li><code>close()</code>: Release the handle.</li>
          </ul>
        </li>
        <li><strong>Context Manager:</strong> Safely scope device usage via <code>with device: ...</code>.</li>
      </ul>

      <h3>GpuMemory</h3>
      <ul>
        <li><code>initialize(pool_bytes=0)</code>: Size the memory pool. 0 defaults to 80% of device memory.</li>
        <li><strong>Properties:</strong> <code>bytes_used</code>, <code>bytes_available</code>, <code>peak_bytes_used</code>.</li>
      </ul>

      <h3>GpuProfiler</h3>
      <ul>
        <li><code>record(kernel, duration_us, work_items=0)</code>: Record a kernel execution manually.</li>
        <li><code>recent_timings(max_count=100) -&gt; list[KernelTiming]</code>: Fetch recent kernel timing data.</li>
        <li><strong>Properties:</strong> <code>total_launches</code>.</li>
        <li><code>clear()</code>: Reset profiler state.</li>
      </ul>

      <h2 id="gpu-config">Configuration & Enums (<code>GpuConfig</code>)</h2>
      <p>
        The <code>GpuConfig</code> struct dictates behavior when selecting devices or building indices.
      </p>
      <ul>
        <li><strong>Fields:</strong> <code>policy</code>, <code>preferred_backend</code>, <code>device_index</code>, <code>build_mode</code>, <code>algorithm</code>, <code>device_memory_pool_mb</code>, <code>pinned_host_pool_mb</code>, <code>fp16_search</code>, <code>unified_memory</code>, <code>batch_window_us</code>, <code>max_batch_size</code>, <code>ef_search</code>, <code>precision</code>, <code>profiling</code>, <code>auto_rebuild_on_startup</code>, <code>rebuild_threshold_ratio</code>, <code>emit_kernel_timings</code>, <code>graph_params</code>, <code>ivf_pq_params</code>.</li>
      </ul>
      <p><strong>Relevant Enums:</strong></p>
      <ul>
        <li><code>GpuPolicy</code>: <code>auto</code>, <code>prefer_gpu</code>, <code>require_gpu</code>, <code>cpu_only</code>, <code>specific</code>.</li>
        <li><code>IndexBuildMode</code>, <code>GpuPrecision</code>, <code>GpuError</code>.</li>
        <li><code>GpuIndexAlgorithm</code>: <code>auto</code>, <code>cagra</code>, <code>ivf_flat</code>, <code>ivf_pq</code>, <code>brute_force</code>.</li>
      </ul>

      <h2 id="modern-accelerator">Modern Accelerator API (<code>Accelerator</code>)</h2>
      <p>
        The modern wrapper significantly reduces boilerplate. The <code>AcceleratorSpec</code> dataclass gives discovery info (<code>name</code>, <code>backend</code>, <code>index</code>, <code>memory_bytes</code>, <code>free_memory_bytes</code>, <code>unified_memory</code>, <code>supports_fp16</code>, <code>raw</code>, <code>memory_gb</code>).
      </p>
      <p>
        The <code>Accelerator</code> class offers:
      </p>
      <ul>
        <li><strong>Properties:</strong> <code>raw</code>, <code>spec</code>, <code>backend</code>, <code>idle</code>, <code>closed</code>.</li>
        <li><code>reserve(pool_bytes=0)</code>: Size the VRAM pool up front.</li>
        <li><code>distances(queries, corpus, *, metric="cosine") -&gt; DistanceMatrix</code>: Pairwise distances.</li>
        <li><code>nearest(distances, *, top) -&gt; TopKResult</code>: Sorting wrapper.</li>
        <li><code>search(queries, corpus, *, top, metric="cosine") -&gt; TopKResult</code>: Fused distance+nearest call.</li>
        <li><code>synchronize()</code>: Host blocking.</li>
        <li><code>memory_usage() -&gt; tuple[int, int, int]</code>: Returns <code>(used, available, peak)</code>.</li>
        <li><code>kernel_timings(limit=100)</code></li>
        <li><code>close()</code> + context manager support.</li>
      </ul>

      <h2 id="brute-force-example">End-to-End Example</h2>
      <h3>Modern API (Preferred)</h3>
      <CodeBlock lang="python">{`import elips

gpu = elips.accelerator()
if gpu is None:
    raise SystemExit("no GPU available")

with gpu:
    gpu.reserve(512 * 1024 * 1024)  # 512 MiB pool
    corpus = [[0.1, 0.2, 0.9], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]
    queries = [[0.1, 0.2, 0.88], [0.85, 0.15, 0.0]]
    
    idx, vals = gpu.search(queries, corpus, top=2, metric="cosine")
    
    for qi, (row_idx, row_vals) in enumerate(zip(idx, vals)):
        print(f"query {qi}: {list(zip(row_idx, row_vals))}")
        
    used, avail, peak = gpu.memory_usage()
    print(f"VRAM used={used}, peak={peak}")
`}</CodeBlock>

      <h3>Low-level Core API</h3>
      <CodeBlock lang="python">{`import elips

device = elips.gpu_select()
if not device:
    raise SystemExit("no GPU available")

with device:
    device.memory.initialize(512 * 1024 * 1024)
    corpus = [[0.1, 0.2, 0.9], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]
    queries = [[0.1, 0.2, 0.88], [0.85, 0.15, 0.0]]
    
    dist_matrix = device.compute_distances(queries, corpus, "cosine")
    top_k_res = device.top_k(dist_matrix, 2)
    
    # Process results ...
`}</CodeBlock>

      <h2 id="batch-stats">Batch Statistics</h2>
      <p>
        <code>BatchStats</code> provides profiling and performance metrics: <code>queries_coalesced</code>, <code>kernel_launches</code>, <code>avg_batch_size</code>, <code>p99_latency_us</code>.
      </p>

      <h2 id="database-integration">Database Integration</h2>
      <p>
        When opening an ELIPS database, you can supply a <code>GpuConfig</code> to offload index building and searching directly.
      </p>
      <CodeBlock lang="python">{`config = elips.GpuConfig(
    policy=elips.GpuPolicy.prefer_gpu,
    algorithm=elips.GpuIndexAlgorithm.cagra,
    fp16_search=True
)

db = elips.open("my_db", gpu_config=config)`}</CodeBlock>

      <h2 id="errors">Error Handling</h2>
      <p>
        The <code>gpu_error_message(error: GpuError) -&gt; str</code> function translates internal enum codes into human-readable strings. Using a closed device handle will raise an exception rather than segfaulting.
      </p>

      <h2 id="pitfalls">Common Pitfalls</h2>
      <ul>
        <li><strong>Raw Memory Operations:</strong> Raw <code>allocate_device</code>, <code>upload</code>, and <code>download</code> are intentionally C++-only. Using incorrect byte counts in C++ leads to silent device memory corruption.</li>
        <li><strong>Use After Free:</strong> Using a <code>close()</code>'d handle raises a Python exception instead of crashing the process.</li>
        <li><strong>OS Restrictions:</strong> Metal backend defaults OFF on non-Apple systems. Do not force it on Linux builds.</li>
        <li><strong>Check for None:</strong> <code>gpu_select()</code> and <code>accelerator()</code> return <code>None</code> on CPU-only machines. Always check the return value.</li>
        <li><strong>Pre-allocation:</strong> Call <code>initialize()</code> or <code>reserve()</code> BEFORE heavy kernel work. Mid-run suballocator growth is slow and causes fragmentation.</li>
        <li><strong>Available Bytes:</strong> <code>bytes_available</code> accurately represents the free list + uncommitted headroom. It does not over-report space.</li>
      </ul>
    </DocsShell>
  );
}
