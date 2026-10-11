import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/guides")({
  head: () => ({
    meta: [
      { title: "Guides — ELIPS Docs" },
      { name: "description", content: "Task-shaped walkthroughs for common ELIPS workflows." },
      { property: "og:title", content: "Guides — ELIPS" },
      { property: "og:description", content: "Task-shaped walkthroughs." },
      { property: "og:url", content: "/docs/guides" },
    ],
    links: [{ rel: "canonical", href: "/docs/guides" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Practice"
      title="Guides"
      toc={[
        { id: "rag", label: "RAG store" },
        { id: "fanout", label: "Shared reader fan-out" },
        { id: "rebuild", label: "Reindex without downtime" },
        { id: "import", label: "Bulk import" },
      ]}
    >
      <p className="text-[18px] text-ink">
        Short, task-shaped walkthroughs. Each one starts from{" "}
        <Link to="/docs">Getting started</Link> and ends with a runnable snippet.
      </p>

      <h2 id="rag">A small RAG store</h2>
      <p>
        Persist chunked documents with lineage, query by text, and surface the original passage on
        every hit.
      </p>
      <CodeBlock lang="python">
        {`import elips

client = elips.PersistentClient("/var/lib/rag", dimension=384, metric="cosine")
docs = client.get_or_create_collection("docs")

for chunk in iter_chunks(corpus):
    docs.add(
        documents=[chunk.text],
        metadatas=[{"source": chunk.source_id, "page": chunk.page}],
    )

client.checkpoint()

for hit in docs.query(query_texts=["how do we rotate keys?"], n_results=5):
    print(hit.text, hit.meta, hit.distance)`}
      </CodeBlock>

      <h2 id="fanout">Shared-reader fan-out</h2>
      <p>
        One writer process refreshes the database; many reader processes serve queries with
        read-only handles.
      </p>
      <CodeBlock lang="python">
        {`# writer.py — exclusive lock
writer = elips.PersistentClient("/srv/index", dimension=768)

# reader.py — shared lock, can run in N processes concurrently
reader = elips.PersistentClient("/srv/index", access_mode="read_only")
result = reader.get_collection("docs").query(
    query_texts=["status of incident 9412"], n_results=10
)`}
      </CodeBlock>

      <h2 id="rebuild">Reindex without downtime</h2>
      <p>
        Build a fresh index in a sibling directory, swap atomically, and let readers pick it up on
        their next open.
      </p>
      <CodeBlock lang="python">
        {`new = elips.PersistentClient("/srv/index.next", dimension=768)
hydrate(new)
new.compact()
new.close()

# atomic swap
os.rename("/srv/index", "/srv/index.prev")
os.rename("/srv/index.next", "/srv/index")`}
      </CodeBlock>

      <h2 id="import">Bulk import</h2>
      <CodeBlock lang="bash">
        {`elips import /srv/index --collection docs --input dump.jsonl --dimension 768
elips checkpoint /srv/index`}
      </CodeBlock>
    </DocsShell>
  );
}
