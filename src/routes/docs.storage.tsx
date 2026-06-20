import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";
import { SketchCard, RecoveryFlowDiagram, OnDiskLayoutDiagram } from "../components/SketchDiagram";

export const Route = createFileRoute("/docs/storage")({
  head: () => ({
    meta: [
      { title: "Storage & recovery — ELIPS Docs" },
      { name: "description", content: "ELIPS on-disk layout, WAL semantics, segmented persistence, checkpoints, and crash recovery." },
      { property: "og:title", content: "Storage & recovery — ELIPS" },
      { property: "og:description", content: "On-disk layout, WAL, segments, checkpoints, and recovery." },
      { property: "og:url", content: "/docs/storage" },
    ],
    links: [{ rel: "canonical", href: "/docs/storage" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell eyebrow="Concepts" title="Storage & recovery" toc={[
      { id: "layout", label: "On disk" },
      { id: "identity", label: "IDENTITY" },
      { id: "embedder", label: "Embedder manifest" },
      { id: "wal", label: "WAL" },
      { id: "checkpoint", label: "Checkpoint & compact" },
      { id: "recovery", label: "Recovery" },
      { id: "ro", label: "Read-only mode" },
    ]}>
      <p className="lede handwritten-lede">
        A persistent ELIPS database is a directory. Each file inside it has
        a single responsibility: identity, WAL, manifest, segments,
        snapshot, lock, embedder.
      </p>

      <h2 id="layout">On disk</h2>
      <SketchCard caption="One directory holds the entire database. Files are independently meaningful and atomically replaced.">
        <OnDiskLayoutDiagram />
      </SketchCard>
      <CodeBlock lang="text" filename="/my_db/">
{`/my_db/
├── LOCK                       # advisory file lock
├── IDENTITY                   # dimension, metric, index type
├── TEXT_EMBEDDER.manifest     # embedder identity
├── wal.log                    # write-ahead log
├── elips.manifest             # segmented mode root
├── text_embedder/
│   └── default_v1_<dim>.localembed
├── segments/
│   └── vault_<n>_<epoch>.segment
└── elips.snapshot             # snapshot mode (compat)`}
      </CodeBlock>

      <h2 id="identity">IDENTITY</h2>
      <p>
        The durable source of truth for <strong>dimension</strong>,{" "}
        <strong>metric</strong>, and <strong>index type</strong>. Existing
        databases reopen with this identity; passing a conflicting value
        raises <code>ConfigError</code>.
      </p>

      <h2 id="embedder">Embedder manifest</h2>
      <p>
        <code>TEXT_EMBEDDER.manifest</code> records provider, model, revision,
        dimension, fingerprint, whether the embedder is rehydratable, and a
        relative artifact path when applicable. For the built-in local
        embedder this manifest plus the <code>.localembed</code> artifact is
        everything required to restore the same embedder on reopen.
      </p>

      <h2 id="wal">WAL</h2>
      <p>
        Every mutation appends to <code>wal.log</code> before the in-memory
        vault changes. Records are framed with a CRC32C. Supported ops:
      </p>
      <ul>
        <li><code>insert</code> — vector + payload.</li>
        <li><code>erase</code> — id-only.</li>
        <li><code>insert_ex</code> — full document attachment, chunk info, and embedding lineage.</li>
      </ul>
      <p>Durability controls when the log flushes:</p>
      <table>
        <thead><tr><th>Mode</th><th>Flush</th></tr></thead>
        <tbody>
          <tr><td><code>paranoid</code></td><td>Flush + fsync per write.</td></tr>
          <tr><td><code>standard</code></td><td>Flush per write.</td></tr>
          <tr><td><code>relaxed</code></td><td>Buffer until checkpoint / close.</td></tr>
          <tr><td><code>ephemeral</code></td><td>No WAL attached.</td></tr>
        </tbody>
      </table>

      <h2 id="checkpoint">Checkpoint &amp; compact</h2>
      <p>
        <code>checkpoint()</code> writes the current logical state and
        truncates the WAL. In segmented mode it writes one fresh segment per
        vault, rewrites <code>elips.manifest</code>, and removes obsolete
        segment files. In snapshot mode it writes{" "}
        <code>elips.snapshot.tmp</code>, then atomically renames into place.
      </p>
      <p>
        <code>compact()</code> rebuilds every vault index from the
        authoritative record store, then checkpoints — useful after large
        deletions or to reset graph topology.
      </p>

      <h2 id="recovery">Recovery</h2>
      <SketchCard caption="Open-time recovery: acquire the lock, resolve identity & embedder, load segments or snapshot, then replay only the valid WAL prefix.">
        <RecoveryFlowDiagram />
      </SketchCard>
      <p>
        Corrupt or truncated WAL tails are tolerated: replay stops at the
        first invalid record and preserves the valid prefix. This is what
        makes ungraceful shutdowns safe in practice.
      </p>

      <h2 id="ro">Read-only mode</h2>
      <p>
        Read-only opens require an existing database and take a shared
        lock. Multiple readers coexist; no WAL writer is attached; every
        mutation path raises <code>StorageError</code>. This is the
        supported mode for fan-out serving and shared-reader analytics.
      </p>
    </DocsShell>
  );
}
