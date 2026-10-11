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
          "Reference for executing ELIPS Query Language (EQL) from Python: $-bindings and the real lowercase grammar — search, get, scan, add, delete.",
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
        { id: "query-methods", label: "query_eql() & native query()" },
        { id: "parameter-bindings", label: "Parameter bindings" },
        { id: "eql-statements", label: "EQL statement syntax" },
        { id: "code-examples", label: "Code examples" },
      ]}
    >
      <p className="text-[18px] text-ink">
        EQL is a small, expression-oriented DSL — not SQL. It has a lexer, a recursive-descent
        parser, an AST, and an executor, and it produces the same <code>Filter</code> and{" "}
        <code>QueryPlan</code> objects the SDK builders produce. You run it from Python with{" "}
        <code>client.query_eql()</code> or directly on the native handle with{" "}
        <code>client.native.query()</code>.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        A statement starts with a verb — <code>search</code>, <code>get</code>, <code>scan</code>,{" "}
        <code>add</code>, or <code>delete</code> — all lowercase and case-sensitive. Query vectors
        are supplied as <code>$name</code> placeholders bound through the <code>bindings</code>{" "}
        argument. See the full grammar on the <Link to="/docs/eql">EQL reference</Link>.
      </p>

      <h2 id="query-methods">query_eql() &amp; native query()</h2>
      <CodeBlock lang="python">{`# High-level client (Client / PersistentClient)
results = client.query_eql(
    'search in documents nearest $q k 10 where category = "tech" yield',
    {"q": query_vector},
)

# Or straight to the native Database handle
results = client.native.query(
    'search in articles nearest $vec k 5 where rating >= 4.5 yield',
    {"vec": [0.1] * 128},
)`}</CodeBlock>

      <h2 id="parameter-bindings">Parameter bindings</h2>
      <p>
        Vector placeholders start with a dollar sign <code>$name</code> and are supplied through the{" "}
        <code>bindings</code> dictionary as float lists or NumPy 1-D vectors. (The old{" "}
        <code>:name</code> colon form never existed in the grammar.)
      </p>
      <CodeBlock lang="python">{`import numpy as np

vec_query = np.random.randn(128).astype(np.float32).tolist()

results = client.query_eql(
    """
    search in product_embeddings
        nearest $target_vec
        k 20
        where price <= 100.0 and in_stock = true
        yield
    """,
    {"target_vec": vec_query},
)`}</CodeBlock>

      <h2 id="eql-statements">EQL statement syntax</h2>
      <ul>
        <li>
          <strong>search:</strong>{" "}
          <code>search in &lt;collection&gt; nearest $vec [k &lt;int&gt;] [threshold &lt;num&gt;] [where &lt;filter&gt;] [rank_by &lt;distance|field&gt;] [project &lt;*|field,...&gt;] yield</code>
        </li>
        <li>
          <strong>get:</strong> <code>get from &lt;collection&gt; id "&lt;string-id&gt;" yield</code>
        </li>
        <li>
          <strong>scan:</strong>{" "}
          <code>scan in &lt;collection&gt; [where &lt;filter&gt;] [offset &lt;int&gt;] [limit &lt;int&gt;] yield</code>
        </li>
        <li>
          <strong>add:</strong>{" "}
          <code>add in &lt;collection&gt; vector [0.1, 0.2, ...] [metadata &#123; "key": value &#125;]</code>
        </li>
        <li>
          <strong>delete:</strong> <code>delete from &lt;collection&gt; id "&lt;string-id&gt;"</code>
        </li>
      </ul>
      <p>
        The pre-1.1 verbs <code>seek</code>/<code>fetch</code>/<code>place</code>/<code>erase</code>{" "}
        are still accepted as deprecated synonyms, and <code>top</code> still works as an alias for{" "}
        <code>k</code>. Text-first retrieval is not exposed through EQL — use{" "}
        <code>collection.query(query_texts=...)</code> for that.
      </p>

      <h2 id="code-examples">Code examples</h2>
      <CodeBlock lang="python">{`import elips

with elips.Client(dimension=4) as client:
    items = client.get_or_create_collection("items")
    items.add(embeddings=[[1.0, 0.0, 0.0, 0.0]], metadatas=[{"status": "active"}])
    items.add(embeddings=[[0.0, 1.0, 0.0, 0.0]], metadatas=[{"status": "archived"}])

    # Execute an EQL query
    hits = client.query_eql(
        'search in items nearest $q k 5 where status = "active" yield',
        {"q": [1.0, 0.0, 0.0, 0.0]},
    )
    print(f"Found active hits: {len(hits)}")`}</CodeBlock>
    </DocsShell>
  );
}
