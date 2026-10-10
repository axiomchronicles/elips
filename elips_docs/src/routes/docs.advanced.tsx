import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/advanced")({
  head: () => ({
    meta: [
      { title: "Advanced patterns — ELIPS Docs" },
      {
        name: "description",
        content:
          "Custom embedders, hybrid fusion, planner introspection, transactions, and shared-reader serving.",
      },
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
    <DocsShell
      eyebrow="Practice"
      title="Advanced patterns"
      toc={[
        { id: "embedder", label: "Custom embedders" },
        { id: "plan", label: "Planner introspection" },
        { id: "txn", label: "Transactions" },
        { id: "lineage", label: "Lineage and migration" },
        { id: "abandon", label: "Recovery testing" },
      ]}
    >
      <h2 id="embedder">Custom embedders</h2>
      <p>
        Two paths exist. The rehydratable local embedder is persisted via{" "}
        <code>TEXT_EMBEDDER.manifest</code> and reattached automatically. A Python callable embedder
        is metadata-only — reopening without the same callable raises <code>ConfigError</code>.
      </p>
      <CodeBlock lang="python">
        {`def my_embedder(texts: list[str]) -> list[list[float]]:
    return model.encode(texts).tolist()

client = elips.PersistentClient(
    "/tmp/elips",
    dimension=384,
    metric="cosine",
    embedder=my_embedder,
)`}
      </CodeBlock>

      <h2 id="plan">Planner introspection</h2>
      <p>
        <code>explain</code> returns the strategy the planner chose, whether
        <code> MetadataIndex</code> narrowed the search, and the candidate count when narrowing
        occurred. Use it in tests to lock query shapes against accidental regressions.
      </p>
      <CodeBlock lang="python">
        {`plan = docs.explain(
    [1.0, 0.0],
    n_results=5,
    where={"kind": "design"},
)
assert plan.metadata_accelerated
assert plan.strategy in {"exact_candidates", "hybrid_fusion"}`}
      </CodeBlock>

      <h2 id="txn">Transactions</h2>
      <p>
        Transactions buffer <code>add</code> and <code>remove</code> in memory and commit
        atomically.
      </p>
      <CodeBlock lang="cpp">
        {`auto txn = db->begin_transaction();
txn.collection("documents").add(elips::Vector{{1.0F, 0.0F}});
txn.collection("documents").add(elips::Vector{{0.0F, 1.0F}});
txn.commit();   // both visible or neither`}
      </CodeBlock>

      <h2 id="lineage">Lineage &amp; migration</h2>
      <p>
        <code>EmbeddingLineage</code> records the provider, model, and revision that produced each
        vector. When you migrate to a new embedder, you can filter on lineage to re-embed only the
        records that need it.
      </p>
      <CodeBlock lang="python">
        {`stale = docs.get(where={"__lineage.model": "bge-small"})
for r in stale:
    docs.delete(ids=[r.key])
    docs.add(text=r.text, metadata=r.meta, id=r.key)`}
      </CodeBlock>

      <h2 id="abandon">Recovery testing</h2>
      <p>
        Calling <code>close()</code> without a preceding <code>checkpoint()</code> leaves pending
        writes in the WAL. Inspect them with <code>client.pending_writes()</code>, then pair with{" "}
        <code>elips verify</code> in integration tests to assert recovery semantics across crash
        points.
      </p>
    </DocsShell>
  );
}
