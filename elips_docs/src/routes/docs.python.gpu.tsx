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
        { id: "backends", label: "Backends & Maturity" },
        { id: "install", label: "Installation & Builds" },
        { id: "has-gpu", label: "Detecting GPU Support" },
        { id: "device-discovery", label: "Device Discovery" },
        { id: "accelerator-spec", label: "AcceleratorSpec" },
        { id: "selecting", label: "Selecting a Device" },
        { id: "native-info", label: "Native Device Info" },
        { id: "bf-knn", label: "GPU Brute-Force kNN" },
        { id: "cpp-types", label: "C++ GPU Types" },
        { id: "pitfalls", label: "Common Pitfalls" },
      ]}
    >
      <p>
        ELIPS can offload distance and top-k computation to the GPU. The current
        acceleration surface is deliberately small: discover devices with{" "}
        <code>elips.accelerators()</code>, pick a backend when you open a client
        with <code>device=</code>, and run exact GPU brute-force search with{" "}
        <code>elips.bf_knn(...)</code>. See the{" "}
        <Link to="/docs/gpu-engine">C++ GPU Engine internals</Link> and the{" "}
        <Link to="/docs/python-sdk">Python SDK Base API</Link> for more context.
      </p>

      <h2 id="backends">Backends &amp; Maturity</h2>
      <p>
        ELIPS ships distance and top-k kernels for the backends below. There are
        no other GPU backends.
      </p>
      <ul>
        <li>
          <strong>Metal</strong> — production. Default ON for Apple Silicon;
          provides the distance and top-k kernels used by GPU search.
        </li>
        <li>
          <strong>CUDA</strong> — experimental. Requires the NVIDIA toolkit.
        </li>
        <li>
          <strong>HIP / ROCm</strong> — experimental. Requires ROCm.
        </li>
        <li>
          <strong>CPU</strong> — always available fallback.
        </li>
      </ul>
      <p>
        GPU kernels cover distance computation and top-k selection only. The
        rest of the index pipeline runs on the CPU.
      </p>

      <h2 id="install">Installation &amp; Builds</h2>
      <ul>
        <li>
          <strong>Metal:</strong> Apple Silicon only. Gated by{" "}
          <code>-DELIPS_GPU_METAL=ON</code> (defaults to ON for Apple
          platforms). Do not force this on Linux.
        </li>
        <li>
          <strong>CUDA:</strong> requires the NVIDIA toolkit.
        </li>
        <li>
          <strong>HIP:</strong> requires ROCm.
        </li>
      </ul>

      <h2 id="has-gpu">Detecting GPU Support</h2>
      <p>
        <code>elips.has_gpu</code> is a boolean that is <code>True</code> when
        the current build was compiled with at least one GPU backend and a
        device is present.
      </p>
      <CodeBlock lang="python">{`import elips

if elips.has_gpu:
    print("GPU acceleration available")
else:
    print("CPU-only build or no device")`}</CodeBlock>

      <h2 id="device-discovery">Device Discovery</h2>
      <ul>
        <li>
          <code>elips.accelerators() -&gt; list[AcceleratorSpec]</code> —
          every GPU detected across the compiled backends.
        </li>
        <li>
          <code>elips.accelerator() -&gt; AcceleratorSpec | None</code> — the
          device ELIPS would pick by default, or <code>None</code> on a
          CPU-only machine.
        </li>
      </ul>
      <CodeBlock lang="python">{`import elips

for spec in elips.accelerators():
    print(spec.backend, spec.index, spec.name, spec.memory_bytes)

best = elips.accelerator()
if best is None:
    print("no GPU available")
else:
    print(f"default device: {best.name} ({best.backend})")`}</CodeBlock>

      <h2 id="accelerator-spec">AcceleratorSpec</h2>
      <p>
        A discovery record describing one GPU. Fields:
      </p>
      <ul>
        <li>
          <code>name</code> (<code>str</code>) — human-readable device name.
        </li>
        <li>
          <code>backend</code> (<code>str</code>) — one of{" "}
          <code>"metal"</code>, <code>"cuda"</code>, <code>"hip"</code>.
        </li>
        <li>
          <code>index</code> (<code>int</code>) — the device ordinal within its
          backend (the <code>N</code> in <code>"cuda:N"</code>).
        </li>
        <li>
          <code>memory_bytes</code> (<code>int</code>) — total device memory.
        </li>
        <li>
          <code>free_memory_bytes</code> (<code>int</code>) — currently free
          device memory.
        </li>
        <li>
          <code>unified_memory</code> (<code>bool</code>) — whether the device
          shares memory with the host (true on Apple Silicon).
        </li>
        <li>
          <code>supports_fp16</code> (<code>bool</code>) — half-precision
          support.
        </li>
        <li>
          <code>raw</code> — the backend-specific raw descriptor.
        </li>
      </ul>

      <h2 id="selecting">Selecting a Device</h2>
      <p>
        Choose a backend when you open a client with the <code>device=</code>{" "}
        keyword. Valid values are <code>"cpu"</code>, <code>"auto"</code>,{" "}
        <code>"metal"</code>, <code>"cuda:N"</code>, and <code>"hip:N"</code>.
        There is <strong>no</strong> <code>gpu=True</code> keyword.
      </p>
      <CodeBlock lang="python">{`import elips

# Let ELIPS choose the best available device
client = elips.Client(dimension=128, device="auto")

# Pin a specific backend/device
client = elips.Client(dimension=128, device="metal")
client = elips.Client(dimension=128, device="cuda:0")`}</CodeBlock>
      <p>
        Requesting a named backend that is absent raises{" "}
        <code>BackendUnavailableError</code> — unless you pass{" "}
        <code>fallback=True</code>, in which case ELIPS silently falls back to
        the CPU.
      </p>
      <CodeBlock lang="python">{`try:
    client = elips.Client(dimension=128, device="cuda:0")
except elips.BackendUnavailableError:
    client = elips.Client(dimension=128, device="cpu")

# Or ask for fallback up front
client = elips.Client(dimension=128, device="cuda:0", fallback=True)`}</CodeBlock>
      <p>
        Advanced device policy can also be supplied as a{" "}
        <code>gpu=GpuConfig(...)</code> object, but the <code>device=</code>{" "}
        string is the recommended selector for almost all use cases.
      </p>

      <h2 id="native-info">Native Device Info</h2>
      <p>
        The high-level client does not expose <code>gpu_info()</code> directly.
        Reach the detailed device report through the native handle:
      </p>
      <CodeBlock lang="python">{`client = elips.Client(dimension=128, device="auto")
info = client.native.gpu_info()
print(info)`}</CodeBlock>

      <h2 id="bf-knn">GPU Brute-Force kNN</h2>
      <p>
        <code>elips.bf_knn(xq, xb, k, metric="l2", device="auto")</code> runs an
        exact brute-force k-nearest-neighbour search with no index build. It
        returns a tuple <code>(D, I)</code> of float32 distances and int64
        labels. <code>device="metal"</code> is the Apple-GPU path; metric
        strings are FAISS-style (<code>"l2"</code> for euclidean). This is
        available in GPU-enabled builds only.
      </p>
      <CodeBlock lang="python">{`import numpy as np, elips

xb = np.random.rand(100000, 128).astype("float32")  # corpus
xq = np.random.rand(1000, 128).astype("float32")    # queries

# Exact GPU brute-force, no index build
D, I = elips.bf_knn(xq, xb, 10, metric="l2", device="metal")
print(D.shape, I.shape)  # (1000, 10) (1000, 10)`}</CodeBlock>
      <p>
        On an Apple M4 (dim=128, 100k corpus, 1k queries, k=10, L2), batched{" "}
        <code>bf_knn</code> on Metal reaches recall 1.000 at roughly 914 QPS;
        single-query latency is far lower (~24 QPS), so batch your queries to
        amortize kernel launch and transfer overhead.
      </p>

      <h2 id="cpp-types">C++ GPU Types</h2>
      <p>
        If you drop into the C++ layer, the GPU abstractions were renamed:{" "}
        <code>GpuBackend</code> (was <code>GpuPort</code>),{" "}
        <code>BackendSelector</code> (was <code>GpuSelector</code>), and{" "}
        <code>QueryBatcher</code> (was <code>DynamicBatcher</code>). See the{" "}
        <Link to="/docs/gpu-engine">GPU Engine internals</Link>.
      </p>

      <h2 id="pitfalls">Common Pitfalls</h2>
      <ul>
        <li>
          <strong>No <code>gpu=True</code>:</strong> select hardware with{" "}
          <code>device="metal"|"cuda:N"|"hip:N"|"auto"|"cpu"</code>, not a
          boolean flag.
        </li>
        <li>
          <strong>Absent backends raise:</strong> a named backend that is not
          present raises <code>BackendUnavailableError</code> unless{" "}
          <code>fallback=True</code> is set.
        </li>
        <li>
          <strong>Batch your queries:</strong> single-query GPU calls are
          dominated by launch and transfer overhead; batch for throughput.
        </li>
        <li>
          <strong>Metal is Apple-only:</strong> the Metal backend defaults OFF
          on non-Apple systems — do not force it on Linux builds.
        </li>
        <li>
          <strong>Check for None:</strong> <code>elips.accelerator()</code>{" "}
          returns <code>None</code> on CPU-only machines.
        </li>
      </ul>
    </DocsShell>
  );
}
