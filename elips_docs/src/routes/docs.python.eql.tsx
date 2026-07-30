import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/eql")({
  head: () => ({
    meta: [
      { title: "Python EQL — Executing Declarative Queries" },
      {
        name: "description",
        content:
          "Complete reference for executing ELIPS Query Language (EQL) in Python: parameter bindings, SEARCH, SCAN, FETCH, INSERT, DELETE, and hybrid projections.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Python EQL Integration"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "query-methods", label: "db.query() & engine.query()" },
        { id: "parameter-bindings", label: "Parameter Bindings" },
        { id: "eql-statements", label: "EQL Statement Syntax" },
        { id: "hybrid-eql", label: "Hybrid EQL Queries" },
        { id: "code-examples", label: "Code Examples" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS Query Language (EQL) brings a declarative, SQL-like query interface to vector search. You can execute EQL statements directly in Python via <code>db.query()</code> or <code>engine.query()</code>.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        EQL allows developers to combine vector similarity search, boolean metadata filtering, pagination, and projection into a single declarative query string with parameterized vector placeholders.
      </p>

      <h2 id="query-methods">db.query() & engine.query()</h2>
      <CodeBlock lang="python">{`# Execution via low-level Database handle
results = db.query(
    "SEARCH FROM documents SEEK :q TOP 10 WHERE category = 'tech'",
    bindings={":q": query_vector}
)

# Execution via modern Engine handle
hits = engine.query(
    "SEARCH FROM articles SEEK :vec TOP 5 WHERE rating >= 4.5",
    bindings={":vec": [0.1] * 1536}
)`}</CodeBlock>

      <h2 id="parameter-bindings">Parameter Bindings</h2>
      <p>
        Vector parameter placeholders start with a colon <code>:name</code> and are supplied via the <code>bindings</code> parameter as a Python dictionary of float lists or NumPy 1D vectors.
      </p>
      <CodeBlock lang="python">{`import numpy as np

vec_query = np.random.randn(384).astype(np.float32).tolist()

hits = engine.query(
    """
    SEARCH FROM product_embeddings
    SEEK :target_vec
    TOP 20
    WHERE price <= 100.0 AND in_stock = true
    """,
    bindings={":target_vec": vec_query}
)`}</CodeBlock>

      <h2 id="eql-statements">EQL Statement Syntax</h2>
      <ul>
        <li>
          <strong>SEARCH:</strong> <code>SEARCH FROM vault_name SEEK :vector TOP k [WHERE filter] [THRESHOLD score]</code>
        </li>
        <li>
          <strong>SCAN:</strong> <code>SCAN FROM vault_name [WHERE filter] [OFFSET o] [LIMIT l]</code>
        </li>
        <li>
          <strong>FETCH:</strong> <code>FETCH FROM vault_name WHERE id = 1042</code>
        </li>
        <li>
          <strong>DELETE:</strong> <code>DELETE FROM vault_name WHERE id = 1042</code>
        </li>
      </ul>

      <h2 id="hybrid-eql">Hybrid EQL Queries</h2>
      <CodeBlock lang="python">{`hits = engine.query(
    """
    SEARCH FROM kb_articles
    SEEK :query_vec
    TEXT 'quantum computing algorithms'
    TOP 10
    WHERE section = 'research'
    HYBRID_WEIGHT 0.3
    """,
    bindings={":query_vec": text_embedding}
)`}</CodeBlock>

      <h2 id="code-examples">Code Examples</h2>
      <CodeBlock lang="python">{`import elips

with elips.connect(":memory:", dimension=4) as engine:
    arena = engine.arena("items")
    arena.write(vector=[1.0, 0.0, 0.0, 0.0], meta={"status": "active"})
    arena.write(vector=[0.0, 1.0, 0.0, 0.0], meta={"status": "archived"})

    # Execute EQL query
    hits = engine.query(
        "SEARCH FROM items SEEK :q TOP 5 WHERE status = 'active'",
        bindings={":q": [1.0, 0.0, 0.0, 0.0]}
    )
    print(f"Found active hits: {len(hits)}")`}</CodeBlock>
    </DocsShell>
  );
}
