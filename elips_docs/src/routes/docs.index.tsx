import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/")({
  head: () => ({
    meta: [
      { title: "Getting started — ELIPS Docs" },
      {
        name: "description",
        content:
          "Install ELIPS, open your first database, and run a text-first query in five minutes.",
      },
      { property: "og:title", content: "Getting started — ELIPS" },
      {
        property: "og:description",
        content:
          "Install ELIPS, open your first database, and run a text-first query in five minutes.",
      },
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
        ELIPS runs inside your process. No daemon, no network hop, no sidecar — a database is a
        directory on disk and an open handle in memory. This page walks the shortest path from{" "}
        <code>cmake</code> to a first text-first query.
      </p>

      <h2 id="install">Install</h2>
      <p>
        Build the core and the Python module from source. The native binary is required even when
        you only use Python — the bindings are a thin layer over the C++23 runtime.
      </p>
      <CodeBlock lang="bash" filename="terminal">
        {`cmake -S . -B build -G Ninja \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DELIPS_BUILD_PYTHON=ON
cmake --build build -j
export PYTHONPATH=$PWD/bindings/python`}
      </CodeBlock>
      <p>
        See <Link to="/docs/installation">Installation</Link> for the full toolchain matrix and
        platform notes.
      </p>

      <h2 id="first-db">Your first database</h2>
      <p>
        An ELIPS database is identified by a path. Use <code>elips.Client</code> for an ephemeral
        in-memory database, or <code>elips.PersistentClient(path)</code> for a persistent one.{" "}
        <code>dimension</code> is the embedding width.
      </p>
      <CodeBlock lang="python">
        {`import elips

client = elips.Client(dimension=128)
docs = client.get_or_create_collection("documents")`}
      </CodeBlock>
      <p>
        New databases attach the built-in local text embedder automatically. Disable that with{" "}
        <code>use_default_text_embedder=False</code> when you bring your own.
      </p>

      <h2 id="ingest">Ingest documents</h2>
      <p>
        Call <code>add()</code> with Chroma-style columns (<code>documents</code>,{" "}
        <code>metadatas</code>, <code>ids</code>), or pass <code>RecordInput</code> objects. Both
        accept text plus typed metadata.
      </p>
      <CodeBlock lang="python">
        {`docs.add(
    documents=["alpha design note", "beta incident runbook"],
    metadatas=[{"kind": "design"}, {"kind": "ops"}],
)`}
      </CodeBlock>

      <h2 id="query">Query</h2>
      <p>
        <code>query(query_texts=...)</code> runs a text-first search; passing both{" "}
        <code>query_embeddings</code> and <code>query_texts</code> fuses vector and text;{" "}
        <code>query(query_embeddings=...)</code> takes a raw vector. Each returns typed{" "}
        <code>Hit</code> rows with distance, metadata, and the original document.
      </p>
      <CodeBlock lang="python">
        {`for hit in docs.query(query_texts="alpha", n_results=2):
    print(hit.key, hit.distance, hit.text, hit.meta)`}
      </CodeBlock>

      <h2 id="next">Where to go next</h2>
      <ul>
        <li>
          <Link to="/docs/configuration">Configuration</Link> — dimension, metric, durability,
          embedders.
        </li>
        <li>
          <Link to="/docs/core-concepts">Core concepts</Link> — the mental model behind
          collections, records, and lineage.
        </li>
        <li>
          <Link to="/docs/architecture">Architecture</Link> — how the planner, indexes, and
          persistence layer compose.
        </li>
        <li>
          <Link to="/docs/eql">EQL</Link> — the declarative query language.
        </li>
      </ul>
    </DocsShell>
  );
}
