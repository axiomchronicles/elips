import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/installation")({
  head: () => ({
    meta: [
      { title: "Installation — ELIPS Docs" },
      { name: "description", content: "Build the ELIPS C++23 core, install the Python bindings, and verify your toolchain." },
      { property: "og:title", content: "Installation — ELIPS" },
      { property: "og:description", content: "Build the ELIPS C++23 core, install the Python bindings, and verify your toolchain." },
      { property: "og:url", content: "/docs/installation" },
    ],
    links: [{ rel: "canonical", href: "/docs/installation" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell eyebrow="Start" title="Installation" toc={[
      { id: "requirements", label: "Requirements" },
      { id: "build", label: "Building from source" },
      { id: "python", label: "Python bindings" },
      { id: "verify", label: "Verify" },
      { id: "gpu", label: "Optional GPU build" },
    ]}>
      <p className="text-[18px] text-ink">
        ELIPS builds with CMake and ships its Python module through
        PyBind11. A single configure + build produces both the C++ static
        library and the Python extension.
      </p>

      <h2 id="requirements">Requirements</h2>
      <ul>
        <li>A C++23 compiler — Clang 16+, GCC 13+, or MSVC 19.36+.</li>
        <li>CMake 3.26 or newer.</li>
        <li>Ninja (recommended) or Make.</li>
        <li>Python 3.10+ for the bindings.</li>
        <li>On Linux, the standard development toolchain (glibc, libstdc++).</li>
      </ul>

      <h2 id="build">Building from source</h2>
      <CodeBlock lang="bash" filename="terminal">
{`git clone https://github.com/axiomchronicles/ellips.git
cd ellips

cmake -S . -B build -G Ninja \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DELIPS_BUILD_PYTHON=ON
cmake --build build -j`}
      </CodeBlock>
      <p>
        The default configuration builds the runtime, the <code>elips</code> CLI,
        and the PyBind11 module. Toggle features through CMake variables:
      </p>
      <table>
        <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th></tr></thead>
        <tbody>
          <tr><td><code>ELIPS_BUILD_PYTHON</code></td><td><code>OFF</code></td><td>Build the Python extension.</td></tr>
          <tr><td><code>ELIPS_BUILD_CLI</code></td><td><code>ON</code></td><td>Build the <code>elips</code> command.</td></tr>
          <tr><td><code>ELIPS_BUILD_TESTS</code></td><td><code>ON</code></td><td>Build the C++ test suite for <code>ctest</code>.</td></tr>
          <tr><td><code>ELIPS_GPU_ENABLED</code></td><td><code>OFF</code></td><td>Compile the GPU index family.</td></tr>
        </tbody>
      </table>

      <h2 id="python">Python bindings</h2>
      <p>
        After the build completes, point Python at the bindings directory:
      </p>
      <CodeBlock lang="bash">
{`export PYTHONPATH=$PWD/bindings/python
python3 -c "import elips; print(elips.__doc__)"`}
      </CodeBlock>
      <p>
        For an installable wheel, use the included <code>setup.py</code> under{" "}
        <code>bindings/python/</code>.
      </p>

      <h2 id="verify">Verify</h2>
      <CodeBlock lang="bash">
{`ctest --test-dir build --output-on-failure
PYTHONPATH=bindings/python python3 tests/python/test_bindings.py`}
      </CodeBlock>

      <h2 id="gpu">Optional GPU build</h2>
      <p>
        The GPU index family lives under <code>src/gpu_engine/</code> and is
        compiled when <code>-DELIPS_GPU_ENABLED=ON</code> is set. Backend
        selection happens at runtime via the device manager; CUDA, Metal,
        and a portable fallback are supported. Domain code only ever talks
        to <code>GpuPort</code>, so the runtime falls back to CPU indexes
        when no GPU is available.
      </p>
    </DocsShell>
  );
}
