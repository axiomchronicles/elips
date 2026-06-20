import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/configuration")({
  head: () => ({
    meta: [
      { title: "Configuration — ELIPS Docs" },
      { name: "description", content: "Dimension, metric, index type, durability, segmented storage, and embedder configuration in ELIPS." },
      { property: "og:title", content: "Configuration — ELIPS" },
      { property: "og:description", content: "All ELIPS configuration knobs in one place." },
      { property: "og:url", content: "/docs/configuration" },
    ],
    links: [{ rel: "canonical", href: "/docs/configuration" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell eyebrow="Start" title="Configuration" toc={[
      { id: "identity", label: "Identity" },
      { id: "indexes", label: "Indexes" },
      { id: "durability", label: "Durability" },
      { id: "storage", label: "Persistence" },
      { id: "embedders", label: "Embedders" },
      { id: "access", label: "Access mode" },
    ]}>
      <p className="text-[18px] text-ink">
        <code>Config</code> is a fluent builder. It is immutable after{" "}
        <code>open()</code>, and a subset of its values become the durable{" "}
        <em>identity</em> of the database, persisted in the <code>IDENTITY</code> file.
      </p>

      <h2 id="identity">Identity (frozen at first open)</h2>
      <p>
        Three values are durable: <strong>dimension</strong>, <strong>metric</strong>, and
        <strong> index type</strong>. Reopening with a conflicting value raises{" "}
        <code>ConfigError</code>.
      </p>
      <CodeBlock lang="python">
{`db = elips.open(
    "/var/lib/myapp/elips",
    dimension=384,
    metric="cosine",       # "cosine" | "euclidean" | "dot"
    index="graph",         # "graph" (HNSW) | "exact"
)`}
      </CodeBlock>

      <h2 id="indexes">Indexes</h2>
      <p>
        <code>graph</code> is the default — a Hierarchical Navigable Small
        World graph backed by <code>HierarchicalGraphIndex</code>. <code>exact</code> performs
        a brute-force scan and is suitable for small vaults, test workloads,
        and ground-truth measurement. Both implement the same{" "}
        <code>IndexPort</code> contract; see{" "}
        <Link to="/docs/algorithms">Algorithms</Link>.
      </p>
      <CodeBlock lang="cpp">
{`auto db = elips::open(
    "/tmp/elips",
    elips::Config{}
        .dimension(128)
        .metric(elips::Metric::cosine)
        .graph_params({.M = 16, .ef_construction = 200, .ef_search = 64}));`}
      </CodeBlock>

      <h2 id="durability">Durability</h2>
      <table>
        <thead><tr><th>Mode</th><th>Behaviour</th></tr></thead>
        <tbody>
          <tr><td><code>paranoid</code></td><td>Flush and <code>fsync</code> on every WAL append.</td></tr>
          <tr><td><code>standard</code></td><td>Flush on every append.</td></tr>
          <tr><td><code>relaxed</code></td><td>Buffer; flush at checkpoint and close.</td></tr>
          <tr><td><code>ephemeral</code></td><td>No WAL is attached. In-memory only.</td></tr>
        </tbody>
      </table>

      <h2 id="storage">Persistence</h2>
      <p>
        Segmented storage is on by default. Disable it with{" "}
        <code>Config::segmented_storage(false)</code> to fall back to the
        single-file <code>elips.snapshot</code> format. Both formats are
        crash-safe; <Link to="/docs/storage">Storage &amp; recovery</Link>{" "}
        walks the on-disk layout.
      </p>

      <h2 id="embedders">Embedders</h2>
      <p>
        New databases auto-attach the built-in local text embedder. Two
        explicit choices are supported:
      </p>
      <ul>
        <li>
          <code>Config.local_text_embedder(...)</code> — a rehydratable
          local embedder. ELIPS restores it automatically on reopen via{" "}
          <code>TEXT_EMBEDDER.manifest</code> and a deterministic artifact.
        </li>
        <li>
          <code>Config.text_embedder(callable, ...)</code> — a Python
          callable. Metadata is persisted, but reopening without providing
          the same callable causes text-first APIs to raise{" "}
          <code>ConfigError</code> rather than silently degrading.
        </li>
      </ul>

      <h2 id="access">Access mode</h2>
      <p>
        <code>read_write</code> takes an exclusive advisory lock — one
        writer per database. <code>read_only</code> takes a shared lock and
        rejects every mutation path with <code>StorageError</code>. Use
        read-only mode for shared-reader analytics or fan-out serving.
      </p>
      <CodeBlock lang="python">
{`reader = elips.open("/var/lib/myapp/elips", access_mode="read_only")
print(reader.vault("documents").seek_text("alpha", top=1)[0].data)`}
      </CodeBlock>
    </DocsShell>
  );
}
