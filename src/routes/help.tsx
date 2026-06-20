import { createFileRoute, Link } from "@tanstack/react-router";
import { StandalonePage } from "../components/Page";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/help")({
  head: () => ({
    meta: [
      { title: "Help center — ELIPS" },
      { name: "description", content: "Troubleshooting guides and diagnostics for common ELIPS issues." },
      { property: "og:title", content: "Help center — ELIPS" },
      { property: "og:description", content: "Troubleshoot common ELIPS issues." },
      { property: "og:url", content: "/help" },
    ],
    links: [{ rel: "canonical", href: "/help" }],
  }),
  component: Page,
});

function Page() {
  return (
    <StandalonePage eyebrow="Support" title="Help center" lede="Quick fixes for the issues that come up first. If your problem is not here, the FAQ and the issue tracker are next.">
      <h2><code>ConfigError</code> on reopen</h2>
      <p>
        The most common cause is a dimension or metric mismatch against the
        on-disk <code>IDENTITY</code>. Inspect it:
      </p>
      <CodeBlock lang="bash">{`elips info /path/to/db`}</CodeBlock>
      <p>
        Pass the values from <code>IDENTITY</code> when calling{" "}
        <code>open()</code>, or omit them entirely — existing databases
        always reopen with their persisted identity. A second cause is a
        Python callable embedder that was not provided on reopen.
      </p>

      <h2><code>LockConflict</code></h2>
      <p>
        Another process holds the write lock on the database directory.
        Find it with <code>lsof &lt;db&gt;/LOCK</code> on Linux/macOS. If the
        previous owner crashed, the OS releases the lock automatically.
      </p>

      <h2>WAL replay aborts</h2>
      <p>
        Run <code>elips verify</code> to print the recovered prefix and
        confirm any tail truncation:
      </p>
      <CodeBlock lang="bash">{`elips verify /path/to/db`}</CodeBlock>

      <h2>Slow first query</h2>
      <p>
        HNSW graphs are warm-loaded on first use. Subsequent queries hit
        the resident graph and are sharply faster. If you serve from cold
        starts often, consider a warm-up query at process start.
      </p>

      <h2>Still stuck?</h2>
      <p>
        Open an <a href="https://github.com/axiomchronicles/ellips/issues" target="_blank" rel="noreferrer">issue</a> with the output of{" "}
        <code>elips info</code>, your config, and a minimal reproduction.
        See also the <Link to="/faq">FAQ</Link>.
      </p>
    </StandalonePage>
  );
}
