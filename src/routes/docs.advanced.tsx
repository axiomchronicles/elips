import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/advanced")({
  head: () => ({
    meta: [
      { title: "Advanced patterns — ELIPS Docs" },
      { name: "description", content: "Custom embedders, hybrid fusion, planner introspection, transactions, and shared-reader serving." },
      { property: "og:title", content: "Advanced patterns — ELIPS" },
      { property: "og:description", content: "Advanced usage patterns." },
      { property: "og:url", content: "/docs/advanced" },
    ],
    links: [{ rel: "canonical", href: "/docs/advanced" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell eyebrow="Practice" title="Advanced patterns" toc={[
      { id: "embedder", label: "Custom embedders" },
      { id: "plan", label: "Planner introspection" },
      { id: "txn", label: "Transactions" },
      { id: "lineage", label: "Lineage and migration" },
      { id: "abandon", label: "Recovery testing" },
    ]}>
      <h2 id="embedder">Custom embedders</h2>
      <p>
        Two paths exist. The rehydratable local embedder is persisted via{" "}
        <code>TEXT_EMBEDDER.manifest</code> and reattached automatically. A
        Python callable embedder is metadata-only — reopening without the
        same callable raises <code>ConfigError</code>.
      </p>
      <CodeBlock lang="python">
{`def my_embedder(texts: list[str]) -> list[list[float]]:
    return model.encode(texts).tolist()

db = elips.open_with_config(
    "/tmp/elips",
    elips.Config()
        .dimension(384)
        .metric("cosine")
        .text_embedder(my_embedder, provider="myco", model="bge-small", revision="v1"),
)`}
      </CodeBlock>

      <h2 id="plan">Planner introspection</h2>
      <p>
        <code>explain_seek</code> returns the strategy the planner chose, whether
        <code> MetadataIndex</code> narrowed the search, and the candidate
        count when narrowing occurred. Use it in tests to lock query
        shapes against accidental regressions.
      </p>
      <CodeBlock lang="python">
{`plan = docs.explain_seek(
    [1.0, 0.0],
    top=5,
    where=elips.Filter().field("kind").equals("design"),
    has_text_component=True,
)
assert plan.metadata_accelerated
assert plan.strategy in {"exact_candidates", "hybrid_fusion"}`}
      </CodeBlock>

      <h2 id="txn">Transactions</h2>
      <p>
        Transactions buffer <code>place</code> and <code>erase</code> in
        memory and commit atomically.
      </p>
      <CodeBlock lang="cpp">
{`auto txn = db->begin_transaction();
txn.vault("documents").place(elips::Vector{{1.0F, 0.0F}});
txn.vault("documents").place(elips::Vector{{0.0F, 1.0F}});
txn.commit();   // both visible or neither`}
      </CodeBlock>

      <h2 id="lineage">Lineage &amp; migration</h2>
      <p>
        <code>EmbeddingLineage</code> records the provider, model, and
        revision that produced each vector. When you migrate to a new
        embedder, you can filter on lineage to re-embed only the records
        that need it.
      </p>
      <CodeBlock lang="python">
{`stale = docs.scan(where=elips.Filter()
                  .field("__lineage.model").equals("bge-small"))
for r in stale:
    docs.erase(r.id)
    docs.place_document(r.document.text, r.payload,
                        lineage=elips.EmbeddingLineage("myco", "bge-large", "v2"))`}
      </CodeBlock>

      <h2 id="abandon">Recovery testing</h2>
      <p>
        <code>abandon()</code> closes the database without checkpointing,
        leaving unfinished recovery work in the WAL. Pair it with{" "}
        <code>elips verify</code> in integration tests to assert recovery
        semantics across crash points.
      </p>
    </DocsShell>
  );
}
