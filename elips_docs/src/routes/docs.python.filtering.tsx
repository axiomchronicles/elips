import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/filtering")({
  head: () => ({
    meta: [
      { title: "Filtering" },
      { name: "description", content: "ELIPS Python Filter API Reference" },
    ],
    links: [],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Filtering"
      toc={[
        { id: "overview", title: "Overview" },
        { id: "fluent-chain", title: "Fluent Chain" },
        { id: "combinators", title: "Combinators" },
        { id: "static-factories", title: "Static Factories" },
        { id: "runtime-eval", title: "Runtime Evaluation" },
        { id: "metadata-acceleration", title: "Metadata Acceleration" },
        { id: "seek-integration", title: "Seek Integration" },
        { id: "explain-seek", title: "Explain Seek" },
        { id: "examples", title: "Examples" },
        { id: "pitfalls", title: "Pitfalls" },
      ]}
    >
      <h2 id="overview">Overview</h2>
      <p>
        The ELIPS Python API provides a powerful and expressive <code>Filter</code> class for querying 
        metadata associated with vectors. It supports fluent chaining of predicates, boolean combinators, 
        and works seamlessly with both vector similarity search (<code>seek</code>) and 
        insertion-order iteration (<code>scan</code>).
      </p>
      <p>
        <strong>Type Aliases used in this API:</strong>
      </p>
      <ul>
        <li><code>MetaValue = Union[bool, int, float, str]</code></li>
        <li><code>PayloadLike = Mapping[str, MetaValue]</code></li>
      </ul>

      <h2 id="fluent-chain">Fluent Chain</h2>
      <p>
        The primary way to construct filters is using the fluent builder pattern. Calling <code>Filter()</code> creates 
        an empty filter that matches everything. You can chain predicates by specifying a field with <code>.field(name)</code> 
        and then an operation. Multiple chained conditions are implicitly <strong>AND-ed</strong> together.
      </p>
      <ul>
        <li><code>.equals(value)</code></li>
        <li><code>.not_equals(value)</code></li>
        <li><code>.lt(value)</code> (less than)</li>
        <li><code>.le(value)</code> (less than or equal)</li>
        <li><code>.gt(value)</code> (greater than)</li>
        <li><code>.gte(value)</code> (greater than or equal)</li>
        <li><code>.one_of(values)</code> (matches any of the values in the list)</li>
        <li><code>.contains(substring)</code> (metadata substring match)</li>
      </ul>
      <CodeBlock lang="python">
{`import elips

# Basic fluent chain (predicates are AND-ed)
f = (
    elips.Filter()
    .field("category").equals("tech")
    .field("score").gte(0.8)
    .field("country").one_of(["US", "GB"])
)`}
      </CodeBlock>

      <h2 id="combinators">Combinators</h2>
      <p>
        For more complex logic, you can combine filters using boolean operators.
      </p>
      <ul>
        <li><code>.and_(other)</code>: Returns a new filter combining this filter and <code>other</code> with an AND operator.</li>
        <li><code>.or_(other)</code>: Returns a new filter combining this filter and <code>other</code> with an OR operator.</li>
        <li><code>Filter.not_(inner)</code>: Static method returning a filter that negates the given <code>inner</code> filter.</li>
      </ul>
      <CodeBlock lang="python">
{`# OR combinator
either = (
    elips.Filter().field("tier").equals("pro")
    .or_(elips.Filter().field("year").gte(2023))
)

# NOT combinator
excluded = elips.Filter.not_(
    elips.Filter().field("status").equals("archived")
)`}
      </CodeBlock>

      <h2 id="static-factories">Static Factories</h2>
      <p>
        In addition to the fluent builder, <code>Filter</code> provides static factories for creating single-predicate filters directly.
      </p>
      <ul>
        <li><code>Filter.compare(field, op, value)</code>: Uses the <code>elips.Comparator</code> enum.</li>
        <li><code>Filter.in_set(field, values)</code>: Equivalent to <code>field().one_of()</code>.</li>
        <li><code>Filter.has_substring(field, substring)</code>: Equivalent to <code>field().contains()</code>.</li>
      </ul>
      <p>The <code>Comparator</code> enum includes: <code>eq</code>, <code>ne</code>, <code>lt</code>, <code>le</code>, <code>gt</code>, <code>ge</code>.</p>
      <CodeBlock lang="python">
{`# Static compare with Comparator enum
f2 = elips.Filter.compare("price", elips.Comparator.lt, 100.0)

# in_set
f3 = elips.Filter.in_set("region", ["eu-west", "us-east"])

# has_substring (full-text substring match on metadata)
f4 = elips.Filter.has_substring("title", "design")`}
      </CodeBlock>

      <h2 id="runtime-eval">Runtime Evaluation</h2>
      <p>
        Filters can be evaluated against regular Python dictionaries without interacting with the database. This is useful for testing or client-side filtering.
      </p>
      <ul>
        <li><code>.matches(payload: dict) -&gt; bool</code>: Tests if the filter matches a dictionary.</li>
        <li><code>.matches_all() -&gt; bool</code>: Returns <code>True</code> if the filter is completely empty (matches anything).</li>
      </ul>
      <CodeBlock lang="python">
{`# Runtime eval without DB
record = {"category": "tech", "score": 0.9, "country": "US"}
print(f.matches(record))  # True

print(elips.Filter().matches_all())  # True`}
      </CodeBlock>

      <h2 id="metadata-acceleration">Metadata Acceleration</h2>
      <p>
        ELIPS utilizes a <code>MetadataIndex</code> to accelerate similarity searches by narrowing down candidates before executing the Approximate Nearest Neighbor (ANN) search. Only equality constraints (<code>.equals()</code> and <code>.one_of()</code> / <code>in_set()</code>) can be used for index pre-filtering. Range predicates (<code>lt</code>, <code>le</code>, <code>gt</code>, <code>ge</code>) and substring matches (<code>contains</code>, <code>has_substring</code>) are applied during a post-filtering phase after ANN candidates are retrieved.
      </p>
      <p>
        You can inspect a filter to see what constraints can be pushed down to the index:
      </p>
      <ul>
        <li><code>.exact_constraints() -&gt; Optional[list[tuple[str, list[MetaValue]]]]</code>: Returns a list of equality constraints if acceleratable, otherwise <code>None</code>.</li>
      </ul>
      <CodeBlock lang="python">
{`# Check accelerability
constraints = f.exact_constraints()
# Returns [("category", ["tech"]), ("country", ["US", "GB"])]
# The gte(0.8) predicate is NOT equality, so it's not in exact_constraints`}
      </CodeBlock>

      <h2 id="seek-integration">Seek Integration</h2>
      <p>
        Filters are heavily used when querying a Vault. Both vector similarity search (<code>seek</code>) and 
        insertion-order iteration (<code>scan</code>) accept a <code>where</code> argument.
      </p>
      <CodeBlock lang="python">
{`# Use in seek
hits = vault.seek([1.0, 0.0], top=10, where=f)

# Use in scan (insertion-order iteration)
rows = vault.scan(where=f, offset=0, limit=100)`}
      </CodeBlock>

      <h2 id="explain-seek">Explain Seek</h2>
      <p>
        To understand how your filter interacts with the query planner, you can use <code>explain_seek()</code>. This returns a <code>QueryPlan</code> object detailing whether metadata acceleration was utilized.
      </p>
      <CodeBlock lang="python">
{`# Explain the plan
plan = vault.explain_seek([1.0, 0.0], top=10, where=f)
print(f"Accelerated: {plan.metadata_accelerated}")
print(f"Strategy: {plan.strategy.name}")`}
      </CodeBlock>

      <h2 id="examples">Examples</h2>
      <p>Here is an example combining complex multi-condition, OR, NOT, and tenant scoping logic:</p>
      <CodeBlock lang="python">
{`tenant_filter = elips.Filter().field("tenant_id").equals("org_123")

active_items = (
    elips.Filter()
    .field("status").equals("active")
    .field("visibility").one_of(["public", "internal"])
)

no_drafts = elips.Filter.not_(
    elips.Filter().field("state").equals("draft")
)

# Combine them all
final_filter = tenant_filter.and_(active_items).and_(no_drafts)`}
      </CodeBlock>

      <h2 id="pitfalls">Pitfalls</h2>
      <ul>
        <li><strong>Chained predicates are AND-ed:</strong> Calling <code>.field(x).equals(y).field(a).equals(b)</code> creates an AND filter. It does NOT overwrite or OR them.</li>
        <li><strong>Empty filters are not errors:</strong> An empty <code>elips.Filter()</code> matches everything. This is usually intended but can be unexpected. Use <code>matches_all()</code> to detect empty filters if you need to validate user input.</li>
        <li><strong>Substring is metadata match:</strong> <code>.contains()</code> and <code>has_substring()</code> perform substring matches on the metadata text, <em>not</em> semantic searches on the document content. Use <code>seek_text</code> or <code>seek_hybrid</code> for semantic text search.</li>
        <li><strong>Immutability in combinators:</strong> <code>.or_()</code> and <code>.and_()</code> return <em>new</em> filters. Calling <code>f.or_(other)</code> does not mutate <code>f</code>.</li>
        <li><strong>Acceleration limitations:</strong> <code>exact_constraints()</code> returns <code>None</code> if ANY predicate in the filter is non-equality (like ranges or substrings). The whole filter will still work correctly (and results will be accurate), but the index cannot accelerate it entirely before the ANN step.</li>
      </ul>
    </DocsShell>
  );
}
