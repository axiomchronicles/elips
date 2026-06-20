import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { SketchCard, TutorialRoadmapDiagram } from "../components/SketchDiagram";
import { lessons, lessonsByLevel, type Level } from "../lib/tutorial";

export const Route = createFileRoute("/docs/tutorial/")({
  head: () => ({
    meta: [
      { title: "Tutorial — ELIPS Docs" },
      {
        name: "description",
        content:
          "Thirty-two hand-drawn lessons across Beginner, Intermediate, and Advanced — the complete ELIPS engine end-to-end.",
      },
      { property: "og:title", content: "ELIPS Tutorial — 32 lessons, three levels" },
      {
        property: "og:description",
        content:
          "From pip install to GPU-accelerated production. Beginner → Intermediate → Advanced, every page grounded in the source.",
      },
      { property: "og:url", content: "/docs/tutorial" },
    ],
    links: [{ rel: "canonical", href: "/docs/tutorial" }],
  }),
  component: TutorialHub,
});

const LEVELS: { level: Level; blurb: string }[] = [
  {
    level: "Beginner",
    blurb: "Install, open a database, write your first record, run your first query.",
  },
  {
    level: "Intermediate",
    blurb:
      "EQL, transactions, durability, embedders, planner observability — the daily-driver surface.",
  },
  {
    level: "Advanced",
    blurb:
      "Storage internals, HNSW, lock manager, query executor, GPU engine — everything under the hood.",
  },
];

function TutorialHub() {
  return (
    <DocsShell eyebrow="Tutorial" title="Learn ELIPS — thirty-two lessons">
      <p className="handwritten-lede text-body">
        From <code>pip install</code> to GPU-accelerated production serving. Three levels,
        thirty-two chapters, every page grounded in the actual source — not paraphrase.
      </p>

      <div className="my-10">
        <SketchCard caption="The full path. Bookmark this page — every lesson links back here.">
          <TutorialRoadmapDiagram />
        </SketchCard>
      </div>

      <p className="text-[14px] text-muted">
        {lessons.length} lessons · grouped by level · linear order works if you read straight
        through.
      </p>

      {LEVELS.map(({ level, blurb }) => {
        const items = lessonsByLevel(level);
        return (
          <section key={level} className="mt-14">
            <div className="flex items-baseline gap-3 mb-1">
              <h2 className="display-md text-ink">{level}</h2>
              <span className="text-[13px] text-muted">{items.length} lessons</span>
            </div>
            <p className="text-[15px] text-body mb-5 max-w-[680px]">{blurb}</p>

            <div className="grid sm:grid-cols-2 gap-4">
              {items.map((l) => (
                <Link
                  key={l.slug}
                  to="/docs/tutorial/$lesson"
                  params={{ lesson: l.slug }}
                  className="block rounded-lg border border-hairline hover:border-ink bg-surface p-5 transition"
                >
                  <div className="flex items-baseline gap-3">
                    <span className="handwritten text-primary" style={{ fontSize: 28 }}>
                      {String(l.number).padStart(2, "0")}
                    </span>
                    <h3 className="text-[17px] font-semibold text-ink">{l.title}</h3>
                  </div>
                  <div className="eyebrow mt-2">{l.eyebrow}</div>
                  <p className="text-[14px] text-body mt-2 leading-relaxed">{l.oneLiner}</p>
                </Link>
              ))}
            </div>
          </section>
        );
      })}

      <p className="text-muted text-[13px] mt-14">
        New here?{" "}
        <Link to="/docs/tutorial/$lesson" params={{ lesson: "01-what-is-elips" }}>
          Start with lesson 1 — What ELIPS is (and isn't)
        </Link>
        .
      </p>
    </DocsShell>
  );
}
