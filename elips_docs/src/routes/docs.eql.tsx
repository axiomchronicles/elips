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
          "EQL grammar — seek, fetch, scan, place, erase — plus filters, projections, and bindings.",
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
        { id: "seek", label: "seek" },
        { id: "fetch", label: "fetch" },
        { id: "scan", label: "scan" },
        { id: "mutate", label: "place / erase" },
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
        {`db.query(
    "seek in docs nearest $q top 10 where year >= 2023 yield",
    bindings={"q": query_vector},
)`}
      </CodeBlock>
      <CodeBlock lang="bash">
        {`elips query /my_db --eql 'seek in docs nearest [0.1,0.2,0.3] top 5 yield'`}
      </CodeBlock>

      <h2 id="seek">seek — nearest-neighbour search</h2>
      <CodeBlock lang="eql">
        {`seek in <vault>
    nearest <[literal] | $binding>
    [top <int>]
    [threshold <number>]
    [where <filter>]
    [rank_by <distance | field>]
    [project <* | field, field, ...>]
    yield`}
      </CodeBlock>
      <ul>
        <li>
          <code>top</code> defaults to 10 — unbounded when a <code>threshold</code> range is
          supplied.
        </li>
        <li>
          <code>rank_by &lt;field&gt;</code> re-sorts results by a metadata field; default is
          distance.
        </li>
        <li>
          <code>project</code> trims the returned payload.
        </li>
      </ul>

      <h2 id="fetch">fetch — by id</h2>
      <CodeBlock lang="eql">{`fetch from <vault> id "<uuid>" yield`}</CodeBlock>

      <h2 id="scan">scan — iterate</h2>
      <CodeBlock lang="eql">{`scan in <vault> [where <filter>] [offset <int>] [limit <int>] yield`}</CodeBlock>

      <h2 id="mutate">place / erase</h2>
      <CodeBlock lang="eql">
        {`place in <vault> vector [0.1, 0.2, ...] [data { "key": value, ... }]
erase from <vault> id "<uuid>"`}
      </CodeBlock>

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
seek in articles
    nearest $embedding
    top 10
    where category = "technology" and published_year >= 2023
    project title, author
    yield

# range search
seek in faces nearest $probe threshold 0.15 where verified = true yield

# membership scan
scan in users where country in ["US", "GB", "CA"] limit 100 yield`}
      </CodeBlock>
    </DocsShell>
  );
}
