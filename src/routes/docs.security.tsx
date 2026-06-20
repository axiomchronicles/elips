import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";

export const Route = createFileRoute("/docs/security")({
  head: () => ({
    meta: [
      { title: "Security & privacy — ELIPS Docs" },
      {
        name: "description",
        content:
          "Local-only data, advisory locking, WAL guarantees, and operational considerations for ELIPS.",
      },
      { property: "og:title", content: "Security & privacy — ELIPS" },
      { property: "og:description", content: "Security and privacy posture." },
      { property: "og:url", content: "/docs/security" },
    ],
    links: [{ rel: "canonical", href: "/docs/security" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell eyebrow="Practice" title="Security & privacy">
      <p className="text-[18px] text-ink">
        ELIPS is embedded. It does not open a network socket, does not spawn a background service,
        and does not phone home. Data lives in the database directory you pass on open.
      </p>

      <h2>Data residency</h2>
      <p>
        Every record — vector, payload, document text, chunk coordinates, lineage — is stored
        locally in the database directory. No data is sent off-host by the runtime. The built-in
        local text embedder runs in-process and reads only the artifact under{" "}
        <code>text_embedder/</code>.
      </p>

      <h2>Process model</h2>
      <p>
        One writer per database directory, enforced by an exclusive advisory lock on{" "}
        <code>LOCK</code>. Shared locks allow multiple read-only handles. Locks are RAII-bound to
        the handle; crashing the process releases the lock through the OS.
      </p>

      <h2>Durability and integrity</h2>
      <p>
        The WAL frames each record with CRC32C and recovery truncates at the first invalid record.
        Checkpoints rename atomically into place. Open-while-being-corrupted scenarios are bounded
        by the on-disk identity in <code>IDENTITY</code>.
      </p>

      <h2>Operational considerations</h2>
      <ul>
        <li>
          The database directory should be treated as sensitive material — back it up the way you
          back up the rest of your application state.
        </li>
        <li>File permissions are inherited from the process; ELIPS does not chmod on its own.</li>
        <li>
          The <code>verify</code> CLI command replays the WAL and validates segments, useful in CI.
        </li>
        <li>
          External Python callable embedders run with full process privileges; vet them like any
          other dependency.
        </li>
      </ul>
    </DocsShell>
  );
}
