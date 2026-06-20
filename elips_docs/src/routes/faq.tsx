import type { ReactNode } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { StandalonePage } from "../components/Page";

export const Route = createFileRoute("/faq")({
  head: () => ({
    meta: [
      { title: "FAQ — ELIPS" },
      {
        name: "description",
        content:
          "Common questions about embedding, deployment, persistence, and operational behaviour in ELIPS.",
      },
      { property: "og:title", content: "FAQ — ELIPS" },
      { property: "og:description", content: "Common questions, answered." },
      { property: "og:url", content: "/faq" },
    ],
    links: [{ rel: "canonical", href: "/faq" }],
    scripts: [
      {
        type: "application/ld+json",
        children: JSON.stringify({
          "@context": "https://schema.org",
          "@type": "FAQPage",
          mainEntity: [
            {
              "@type": "Question",
              name: "Is ELIPS a database server?",
              acceptedAnswer: {
                "@type": "Answer",
                text: "No. ELIPS is an embedded engine. A database is a directory on disk and an open handle in your process. There is no daemon to install and no network port to open.",
              },
            },
            {
              "@type": "Question",
              name: "Can multiple processes share a database?",
              acceptedAnswer: {
                "@type": "Answer",
                text: "One writer at a time, many readers. The writer takes an exclusive advisory file lock; readers take shared locks via access_mode=\"read_only\".",
              },
            },
            {
              "@type": "Question",
              name: "What embeddings does ELIPS use?",
              acceptedAnswer: {
                "@type": "Answer",
                text: "New databases attach a built-in local text embedder automatically. You can attach a rehydratable local embedder, a custom Python callable, or disable auto-attach entirely.",
              },
            },
            {
              "@type": "Question",
              name: "What happens on a crash?",
              acceptedAnswer: {
                "@type": "Answer",
                text: "The WAL is replayed on the next open. Each record is CRC32C-framed; replay truncates at the first invalid record and preserves the valid prefix.",
              },
            },
            {
              "@type": "Question",
              name: "How big can a vault get?",
              acceptedAnswer: {
                "@type": "Answer",
                text: "ELIPS is designed for the embedded scale — millions of vectors comfortably, larger with segmented storage and HNSW tuned for memory budget. There is no hard ceiling baked into the format.",
              },
            },
            {
              "@type": "Question",
              name: "Does ELIPS require a GPU?",
              acceptedAnswer: {
                "@type": "Answer",
                text: "No. The default build is CPU-only. Compile with -DELIPS_GPU_ENABLED=ON to opt in to the GPU index family.",
              },
            },
            {
              "@type": "Question",
              name: "Why C++23?",
              acceptedAnswer: {
                "@type": "Answer",
                text: "std::span, concepts, designated initialisers, and structured bindings let the public surface stay precisely typed without overhead.",
              },
            },
            {
              "@type": "Question",
              name: "Where can I report a bug?",
              acceptedAnswer: {
                "@type": "Answer",
                text: "Open an issue on the GitHub tracker. Include the output of elips info and elips verify when relevant.",
              },
            },
          ],
        }),
      },
    ],
  }),
  component: Page,
});

const FAQS: [string, ReactNode][] = [
  [
    "Is ELIPS a database server?",
    <p>
      No. ELIPS is embedded. A database is a directory on disk and an open handle in your process.
      There is no daemon to install and no network port to open.
    </p>,
  ],
  [
    "Can multiple processes share a database?",
    <p>
      One writer at a time, many readers. The writer takes an exclusive advisory file lock; readers
      take shared locks via <code>access_mode="read_only"</code>.
    </p>,
  ],
  [
    "What embeddings does ELIPS use?",
    <p>
      New databases attach a built-in local text embedder automatically. You can attach a
      rehydratable local embedder, a custom Python callable, or disable auto-attach entirely.
    </p>,
  ],
  [
    "What happens on a crash?",
    <p>
      The WAL is replayed on the next open. Each record is CRC32C-framed; replay truncates at the
      first invalid record and preserves the valid prefix.
    </p>,
  ],
  [
    "How big can a vault get?",
    <p>
      ELIPS is designed for the embedded scale — millions of vectors comfortably, larger with
      segmented storage and HNSW tuned for memory budget. There is no hard ceiling baked into the
      format.
    </p>,
  ],
  [
    "Does ELIPS require a GPU?",
    <p>
      No. The default build is CPU-only. Compile with <code>-DELIPS_GPU_ENABLED=ON</code> to opt in
      to the GPU index family.
    </p>,
  ],
  [
    "Why C++23?",
    <p>
      <Link to="/docs/design-decisions">ADR-0001</Link>. <code>std::span</code>, concepts,
      designated initialisers, and structured bindings let the public surface stay precisely typed
      without overhead.
    </p>,
  ],
  [
    "Where can I report a bug?",
    <p>
      Open an issue on the{" "}
      <a href="https://github.com/axiomchronicles/ellips/issues" target="_blank" rel="noreferrer">
        GitHub tracker
      </a>
      . Include the output of <code>elips info</code> and <code>elips verify</code> when relevant.
    </p>,
  ],
];

function Page() {
  return (
    <StandalonePage
      eyebrow="Project"
      title="Frequently asked"
      lede="Answers to the questions that come up most often — pulled from the code, not the marketing."
    >
      <div className="not-prose space-y-3">
        {FAQS.map(([q, a]) => (
          <details
            key={q}
            className="group border border-hairline rounded-lg bg-surface open:bg-canvas-soft transition"
          >
            <summary className="cursor-pointer list-none px-5 py-4 flex items-center justify-between text-ink text-[15px] font-medium">
              <span>{q}</span>
              <span className="text-muted text-xl transition group-open:rotate-45">+</span>
            </summary>
            <div className="px-5 pb-5 -mt-1 text-body text-[14.5px] leading-relaxed prose">{a}</div>
          </details>
        ))}
      </div>
    </StandalonePage>
  );
}
