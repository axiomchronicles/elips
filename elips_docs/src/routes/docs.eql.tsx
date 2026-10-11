import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/eql")({
  head: () => ({
    meta: [
      { title: "EQL — ELIPS Query Language" },
      {
        name: "description",
        content:
          "EQL grammar — search, get, scan, add, delete — plus filters, projections, and bindings.",
      },
      { property: "og:title", content: "EQL — ELIPS Query Language" },
      { property: "og:description", content: "ELIPS Query Language reference." },
      { property: "og:url", content: "/docs/eql" },
    ],
    links: [{ rel: "canonical", href: "/docs/eql" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference"
      title="EQL"
      toc={[
        { id: "shape", label: "Shape" },
        { id: "search", label: "search" },
        { id: "get", label: "get" },
        { id: "scan", label: "scan" },
        { id: "mutate", label: "add / delete" },
        { id: "filters", label: "Filters" },
        { id: "examples", label: "Examples" },
      ]}
    >
      <p className="text-[18px] text-ink">
        EQL is a small expression-oriented query language with a lexer, recursive-descent parser,
        AST, and executor. It produces the same <code>Filter</code> and <code>QueryPlan</code>{" "}
        objects the SDK builders produce.
      </p>

      <h2 id="shape">Shape</h2>
      <CodeBlock lang="python">
        {`client.query_eql(
    "search in docs nearest $q k 10 where year >= 2023 yield",
    {"q": query_vector},
)`}
      </CodeBlock>
      <CodeBlock lang="bash">
        {`elips query /my_db --eql 'search in docs nearest [0.1,0.2,0.3] k 5 yield'`}
      </CodeBlock>

      <h2 id="search">search — nearest-neighbour search</h2>
      <CodeBlock lang="eql">
        {`search in <collection>
    nearest <[literal] | $binding>
    [k <int>]
    [threshold <number>]
    [where <filter>]
    [rank_by <distance | field>]
    [project <* | field, field, ...>]
    yield`}
      </CodeBlock>
      <ul>
        <li>
          <code>k</code> defaults to 10 — unbounded when a <code>threshold</code> range is
          supplied. <code>top</code> is still accepted as a deprecated alias for <code>k</code>.
        </li>
        <li>
          <code>rank_by &lt;field&gt;</code> re-sorts results by a metadata field; default is
          distance.
        </li>
        <li>
          <code>project</code> trims the returned payload.
        </li>
      </ul>

      <h2 id="get">get — by id</h2>
      <CodeBlock lang="eql">{`get from <collection> id "<uuid>" yield`}</CodeBlock>

      <h2 id="scan">scan — iterate</h2>
      <CodeBlock lang="eql">{`scan in <collection> [where <filter>] [offset <int>] [limit <int>] yield`}</CodeBlock>

      <h2 id="mutate">add / delete</h2>
      <CodeBlock lang="eql">
        {`add in <collection> vector [0.1, 0.2, ...] [metadata { "key": value, ... }]
delete from <collection> id "<uuid>"`}
      </CodeBlock>
      <p>
        The verbs are lowercase and case-sensitive. The pre-1.1 spellings{" "}
        <code>seek</code>/<code>fetch</code>/<code>place</code>/<code>erase</code> remain accepted as
        deprecated synonyms.
      </p>

      <h2 id="filters">Filters</h2>
      <CodeBlock lang="eql">
        {`filter     := term (("and" | "or") term)* | "not" filter | "(" filter ")"
term       := field <comparator> value
            | field "in" [ value, value, ... ]
            | field "contains" "substring"
comparator := = | != | < | <= | > | >=
value      := "string" | number | true | false`}
      </CodeBlock>
      <p>
        Numbers without a decimal point are integers. <code>contains</code> is a substring match on
        string fields. Comments start with <code>#</code> and run to end of line.
      </p>

      <h2 id="examples">Examples</h2>
      <CodeBlock lang="eql">
        {`# filtered search with projection
search in articles
    nearest $embedding
    k 10
    where category = "technology" and published_year >= 2023
    project title, author
    yield

# range search
search in faces nearest $probe threshold 0.15 where verified = true yield

# membership scan
scan in users where country in ["US", "GB", "CA"] limit 100 yield`}
      </CodeBlock>
    </DocsShell>
  );
}
