import { createFileRoute, Link, notFound } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { LessonBlocks } from "../components/LessonRenderer";
import { lessonBySlug, neighbors } from "../lib/tutorial";

export const Route = createFileRoute("/docs/tutorial/$lesson")({
  loader: ({ params }) => {
    const lesson = lessonBySlug(params.lesson);
    if (!lesson) throw notFound();
    return { lesson };
  },
  head: ({ loaderData }) => {
    const l = loaderData?.lesson;
    const title = l ? `${l.title} · Lesson ${l.number} — ELIPS Tutorial` : "ELIPS Tutorial";
    const desc = l?.oneLiner ?? "ELIPS tutorial lesson.";
    return {
      meta: [
        { title },
        { name: "description", content: desc },
        { property: "og:title", content: title },
        { property: "og:description", content: desc },
      ],
    };
  },
  component: LessonPage,
});

function LessonPage() {
  const { lesson } = Route.useLoaderData();
  const { prev, next } = neighbors(lesson.slug);

  return (
    <DocsShell eyebrow={lesson.eyebrow} title={lesson.title}>
      <div className="flex items-center gap-3 mb-2">
        <span className="handwritten text-primary" style={{ fontSize: 36 }}>
          Lesson {String(lesson.number).padStart(2, "0")}
        </span>
        <Link to="/docs/tutorial" className="text-[13px] text-muted hover:text-ink">
          ← back to roadmap
        </Link>
      </div>

      <LessonBlocks blocks={lesson.blocks} />

      <nav className="mt-16 pt-8 border-t border-hairline grid grid-cols-2 gap-4">
        <div>
          {prev && (
            <Link
              to="/docs/tutorial/$lesson"
              params={{ lesson: prev.slug }}
              className="block rounded-lg border border-hairline p-4 hover:border-ink transition"
            >
              <div className="eyebrow">← Previous</div>
              <div className="text-ink font-semibold mt-1">{prev.title}</div>
            </Link>
          )}
        </div>
        <div>
          {next && (
            <Link
              to="/docs/tutorial/$lesson"
              params={{ lesson: next.slug }}
              className="block rounded-lg border border-hairline p-4 hover:border-ink transition text-right"
            >
              <div className="eyebrow">Next →</div>
              <div className="text-ink font-semibold mt-1">{next.title}</div>
            </Link>
          )}
        </div>
      </nav>
    </DocsShell>
  );
}
