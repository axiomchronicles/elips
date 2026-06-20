import { createFileRoute } from "@tanstack/react-router";
import { StandalonePage } from "../components/Page";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/contributing")({
  head: () => ({
    meta: [
      { title: "Contributing — ELIPS" },
      {
        name: "description",
        content: "Coding standards, testing matrix, and the release process for the ELIPS project.",
      },
      { property: "og:title", content: "Contributing — ELIPS" },
      { property: "og:description", content: "How to contribute to ELIPS." },
      { property: "og:url", content: "/contributing" },
    ],
    links: [{ rel: "canonical", href: "/contributing" }],
  }),
  component: Page,
});

function Page() {
  return (
    <StandalonePage
      eyebrow="Project"
      title="Contributing"
      lede="ELIPS is an open project. Patches that improve correctness, performance, or clarity are welcome — including documentation patches against this site."
    >
      <h2>Coding standards</h2>
      <p>
        The C++ core targets the{" "}
        <a
          href="https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines"
          target="_blank"
          rel="noreferrer"
        >
          C++ Core Guidelines
        </a>
        . Domain code depends on ports (<code>IndexPort</code>, <code>GpuPort</code>,
        <code>TextEmbedderPort</code>), never on concrete engines. Backend headers stay inside their
        subsystem directory.
      </p>
      <ul>
        <li>
          RAII for every resource. Manual <code>close()</code> calls are a smell.
        </li>
        <li>
          Scoped enums (<code>enum class</code>) with explicit underlying types where wire-stable.
        </li>
        <li>
          Value-typed data containers (<code>Vector</code>, <code>RecordID</code>,{" "}
          <code>Filter</code>).
        </li>
        <li>
          Virtual destructors and <code>override</code> / <code>final</code> on every port
          implementation.
        </li>
      </ul>

      <h2>Tests</h2>
      <CodeBlock lang="bash">
        {`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DELIPS_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
PYTHONPATH=bindings/python python3 tests/python/test_bindings.py`}
      </CodeBlock>
      <p>
        The test tree under <code>tests/</code> covers core behaviour, concurrency (multi-reader and
        single-writer), integration with the local text embedder, and the Python binding surface.
        New features need at least one test, ideally in both C++ and Python.
      </p>

      <h2>Patch shape</h2>
      <ol>
        <li>Open or comment on an issue describing the change.</li>
        <li>
          Branch from <code>main</code>; keep the patch focused.
        </li>
        <li>
          Run <code>ctest</code> and the Python suite locally; CI runs both.
        </li>
        <li>
          Update <code>docs/</code> in the same PR when surface changes.
        </li>
        <li>Open a pull request with the issue link and a brief rationale.</li>
      </ol>

      <h2>Release process</h2>
      <p>
        Releases follow semantic versioning. The release process is documented at{" "}
        <code>docs/developer-guide/release-process.md</code> in the repository: tag → build
        artifacts → publish Python wheel → update changelog.
      </p>
    </StandalonePage>
  );
}
