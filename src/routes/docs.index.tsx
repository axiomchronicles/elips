import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/")({
  head: () => ({
    meta: [
      { title: "Getting started — ELIPS Docs" },
      { name: "description", content: "Install ELIPS, open your first database, and run a text-first query in five minutes." },
      { property: "og:title", content: "Getting started — ELIPS" },
      { property: "og:description", content: "Install ELIPS, open your first database, and run a text-first query in five minutes." },
      { property: "og:url", content: "/docs" },
    ],
    links: [{ rel: "canonical", href: "/docs" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Start · 5 minutes"
      title="Getting started"
      toc={[
        { id: "install", label: "Install" },
        { id: "first-db", label: "Your first database" },
        { id: "ingest", label: "Ingest documents" },
        { id: "query", label: "Query" },
        { id: "next", label: "Where to go next" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS runs inside your process. No daemon, no network hop, no
        sidecar — a database is a directory on disk and an open handle in
        memory. This page walks the shortest path from <code>cmake</code> to a
        first text-first query.
      </p>

      <h2 id="install">Install</h2>
      <p>
        Build the core and the Python module from source. The native binary
        is required even when you only use Python — the bindings are a thin
        layer over the C++23 runtime.
      </p>
      <CodeBlock lang="bash" filename="terminal">
{`cmake -S . -B build -G Ninja \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DELIPS_BUILD_PYTHON=ON
cmake --build build -j
export PYTHONPATH=$PWD/bindings/python`}
      </CodeBlock>
      <p>
        See <Link to="/docs/installation">Installation</Link> for the full
        toolchain matrix and platform notes.
      </p>

      <h2 id="first-db">Your first database</h2>
      <p>
        An ELIPS database is identified by a path. Pass <code>":memory:"</code> for an ephemeral
        database, or a directory path for a persistent one. The first
        argument to <code>connect</code> is the path; <code>dimension</code> is
        the embedding width.
      </p>
      <CodeBlock lang="python">
{`import elips

engine = elips.connect(":memory:", dimension=128)
arena = engine.arena("documents")`}
      </CodeBlock>
      <p>
        New databases attach the built-in local text embedder automatically.
        Disable that with <code>use_default_text_embedder=False</code> when
        you bring your own.
      </p>

      <h2 id="ingest">Ingest documents</h2>
      <p>
        Use <code>RecordInput</code> with the modern wrapper, or call
        <code>place_document()</code> on a vault for the low-level API.
        Both accept text plus typed metadata.
      </p>
      <CodeBlock lang="python">
{`arena.write_many([
    elips.RecordInput(text="alpha design note", meta={"kind": "design"}),
    elips.RecordInput(text="beta incident runbook", meta={"kind": "ops"}),
])`}
      </CodeBlock>

      <h2 id="query">Query</h2>
      <p>
        <code>probe_text</code> runs a text-first search; <code>probe_hybrid</code> fuses
        a vector with text; <code>probe</code> takes a raw vector. Each
        returns typed <code>Hit</code> rows with distance, payload, and the
        original document.
      </p>
      <CodeBlock lang="python">
{`for hit in arena.probe_text("alpha", top=2):
    print(hit.key, hit.distance, hit.text, hit.meta)`}
      </CodeBlock>

      <h2 id="next">Where to go next</h2>
      <ul>
        <li><Link to="/docs/configuration">Configuration</Link> — dimension, metric, durability, embedders.</li>
        <li><Link to="/docs/core-concepts">Core concepts</Link> — the mental model behind vaults, records, and lineage.</li>
        <li><Link to="/docs/architecture">Architecture</Link> — how the planner, indexes, and persistence layer compose.</li>
        <li><Link to="/docs/eql">EQL</Link> — the declarative query language.</li>
      </ul>
    </DocsShell>
  );
}
