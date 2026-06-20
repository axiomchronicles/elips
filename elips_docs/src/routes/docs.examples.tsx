import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/examples")({
  head: () => ({
    meta: [
      { title: "Examples — ELIPS Docs" },
      {
        name: "description",
        content: "Concrete Python and C++ snippets straight from the ELIPS repository.",
      },
      { property: "og:title", content: "Examples — ELIPS" },
      { property: "og:description", content: "Concrete usage examples." },
      { property: "og:url", content: "/docs/examples" },
    ],
    links: [{ rel: "canonical", href: "/docs/examples" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell eyebrow="Practice" title="Examples">
      <p className="text-[18px] text-ink">
        Drawn directly from <code>examples/python/</code> and <code>examples/cpp/</code> in the
        repository.
      </p>

      <h2>Python — getting started</h2>
      <CodeBlock lang="python" filename="examples/python/01_getting_started.py">
        {`import elips

engine = elips.connect(":memory:", dimension=128)
arena = engine.arena("documents")
arena.ingest(
    texts=["alpha design note", "beta incident runbook"],
    meta=[{"kind": "design"}, {"kind": "ops"}],
)

for hit in arena.probe_text("alpha", top=2):
    print(hit.key, hit.distance, hit.text, hit.meta)`}
      </CodeBlock>

      <h2>C++ — getting started</h2>
      <CodeBlock lang="cpp" filename="examples/cpp/01_getting_started.cpp">
        {`#include "elips/elips.hpp"
#include <print>

int main() {
    auto db = elips::open(
        ":memory:",
        elips::Config{}.dimension(128).metric(elips::Metric::cosine));

    auto& docs = db->vault("documents");
    docs.place_document("alpha design note", {{"kind", std::string{"design"}}});
    docs.place_document("beta incident runbook", {{"kind", std::string{"ops"}}});

    for (auto const& hit : docs.seek_text("alpha", 2)) {
        std::println("{} {} {}", hit.id.to_string(), hit.distance,
                     hit.document->text);
    }
}`}
      </CodeBlock>

      <h2>EQL from the SDK</h2>
      <CodeBlock lang="python">
        {`rows = db.query(
    "seek in documents nearest $q top 10 "
    "where kind = \\"design\\" project kind yield",
    bindings={"q": query_vector},
)`}
      </CodeBlock>

      <h2>Hybrid retrieval</h2>
      <CodeBlock lang="python">
        {`hits = docs.seek_hybrid(
    vector=query_vector,
    text="rotate KMS keys",
    top=5,
    where=elips.Filter().field("kind").equals("runbook"),
)`}
      </CodeBlock>
    </DocsShell>
  );
}
