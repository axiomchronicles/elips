import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/cli")({
  head: () => ({
    meta: [
      { title: "CLI — ELIPS Docs" },
      {
        name: "description",
        content:
          "The elips command — info, collections, stats, verify, query, checkpoint, import, export, bench, quantize.",
      },
      { property: "og:title", content: "CLI — ELIPS" },
      { property: "og:description", content: "elips command reference." },
      { property: "og:url", content: "/docs/cli" },
    ],
    links: [{ rel: "canonical", href: "/docs/cli" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell eyebrow="Reference" title="CLI">
      <p className="text-[18px] text-ink">
        <code>elips</code> is a local command-line tool for inspection, maintenance, and EQL
        execution. No server. The build produces an <code>elips</code> binary you can drop on your{" "}
        <code>PATH</code>.
      </p>

      <h2>Shape</h2>
      <CodeBlock lang="bash">{`elips <command> <db_path> [options]`}</CodeBlock>
      <p>
        When creating a new database, pass <code>--dimension</code> (and optionally{" "}
        <code>--metric</code>, <code>--index</code>). Existing databases read their identity from
        disk.
      </p>

      <h2>Inspect</h2>
      <CodeBlock lang="bash">
        {`elips info /my_db
elips collections /my_db
elips stats /my_db
elips verify /my_db          # replay WAL + validate; prints OK / CORRUPT`}
      </CodeBlock>

      <h2>Query</h2>
      <CodeBlock lang="bash">
        {`elips query /my_db --dimension 3 \\
  --eql 'add in docs vector [0.1,0.2,0.3] metadata {"t":"a"}'

elips query /my_db --eql 'search in docs nearest [0.1,0.2,0.3] k 5 project t yield'
elips query /my_db --file query.eql`}
      </CodeBlock>

      <h2>Maintenance</h2>
      <CodeBlock lang="bash">
        {`elips checkpoint /my_db
elips quantize /my_db --collection docs --codec sq8`}
      </CodeBlock>

      <h2>Import / Export</h2>
      <p>
        JSON Lines, one record per line: <code>{`{"key","vector","meta"}`}</code>.
      </p>
      <CodeBlock lang="bash">
        {`elips export /my_db --collection docs --output docs.jsonl
elips import /my_db --collection docs --input docs.jsonl --dimension 3`}
      </CodeBlock>

      <h2>Benchmark</h2>
      <CodeBlock lang="bash">{`elips bench /tmp/bench_db --count 100000 --dim 768`}</CodeBlock>

      <h2>Output</h2>
      <p>
        <code>query</code> prints one JSON object per result row:
      </p>
      <CodeBlock lang="json">{`{"key":"019e...","distance":0.0061,"meta":{"t":"a"}}`}</CodeBlock>
    </DocsShell>
  );
}
