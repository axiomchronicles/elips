import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";

export const Route = createFileRoute("/docs/project-history")({
  head: () => ({
    meta: [
      { title: "Project history — ELIPS Docs" },
      {
        name: "description",
        content:
          "Why ELIPS exists, where it sits in the vector-database landscape, and what v1.0 set out to prove.",
      },
      { property: "og:title", content: "Project history — ELIPS" },
      {
        property: "og:description",
        content:
          "Origins, motivation, and v1.0 scope of the Embedded Local Index & Persistence System.",
      },
      { property: "og:url", content: "/docs/project-history" },
    ],
    links: [{ rel: "canonical", href: "/docs/project-history" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Overview"
      title="Project history"
      toc={[
        { id: "origins", label: "Origins" },
        { id: "motivation", label: "Motivation" },
        { id: "scope", label: "v1.0 scope" },
        { id: "name", label: "The name" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS — <strong>E</strong>mbedded <strong>L</strong>ocal <strong>I</strong>ndex &amp;{" "}
        <strong>P</strong>ersistence <strong>S</strong>ystem — was conceived in 2024 to answer one
        question: what would a vector database look like if it were built like SQLite? Embedded,
        in-process, zero infrastructure.
      </p>

      <h2 id="origins">Origins: built from first principles</h2>
      <p>
        The core was implemented from first principles in C++23. No third-party vector-search
        libraries (FAISS, hnswlib, nmslib) are used for the core indexing. Every subsystem — the
        vector type, the distance kernels, the HNSW graph, the WAL, the query language, the Python
        bindings — was written for ELIPS. The only external dependencies are the C++23 standard
        library, GoogleTest (test-only), PyBind11 (Python bindings), and optional GPU backend
        libraries (cuVS, MPS, oneMKL).
      </p>
      <p>
        Per <Link to="/docs/design-decisions">ADR-0001</Link>, the decision to use C++23 was driven
        by the need for predictable, allocation-controlled performance for SIMD distance kernels and
        graph traversal, plus first-class embeddability in both C++ and Python processes with no
        language runtime beyond the standard library.
      </p>

      <h2 id="motivation">Motivation</h2>
      <p>
        In 2024 the vector-database landscape was dominated by client-server architectures — cloud
        APIs, server processes, REST/gRPC/GraphQL endpoints, deployment pipelines. Powerful for
        large teams; disproportionate for CLIs, desktop apps, research notebooks, CI pipelines, edge
        devices, and single-process services. The friction of "spin up a vector database"
        discourages adoption of semantic search in smaller projects.
      </p>
      <p>
        ELIPS targets that gap: <strong>vector search as a library, not a service</strong>. If
        SQLite replaced client-server RDBMSes for embedded use cases, ELIPS aims to do the same for
        vector search.
      </p>

      <h2 id="scope">v1.0 scope</h2>
      <p>
        Version 1.0 ships a vertically complete prototype: every layer from the domain types through
        the WAL, planner, EQL, CLI, and Python bindings. The deferred list under{" "}
        <Link to="/docs/roadmap">Roadmap</Link> is intentional — each future capability has a v1.0
        seam (a port, a factory, a dispatch table) so it can land additively without breaking the
        existing surface.
      </p>

      <h2 id="name">The name</h2>
      <p>
        The four pillars are baked into the name: <strong>embedded</strong> (in-process),{" "}
        <strong>local</strong> (disk on the host), <strong>indexed</strong> (HNSW for fast search),
        and <strong>persistent</strong> (WAL plus snapshot or segmented manifest).
      </p>
    </DocsShell>
  );
}
