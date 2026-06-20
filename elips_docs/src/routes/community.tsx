import { createFileRoute, Link } from "@tanstack/react-router";
import { StandalonePage } from "../components/Page";

export const Route = createFileRoute("/community")({
  head: () => ({
    meta: [
      { title: "Community — ELIPS" },
      {
        name: "description",
        content: "Where ELIPS contributors meet — GitHub discussions, issues, and code review.",
      },
      { property: "og:title", content: "Community — ELIPS" },
      { property: "og:description", content: "Where ELIPS contributors meet." },
      { property: "og:url", content: "/community" },
    ],
    links: [{ rel: "canonical", href: "/community" }],
  }),
  component: Page,
});

function Page() {
  return (
    <StandalonePage
      eyebrow="Project"
      title="Community"
      lede="ELIPS is built in the open on GitHub. Code review, issue triage, and design discussion all happen there."
    >
      <div className="not-prose grid sm:grid-cols-2 gap-4 my-4">
        {[
          {
            h: "GitHub repository",
            d: "Source, issues, pull requests, and the ADR log under docs/adr/.",
            href: "https://github.com/axiomchronicles/ellips",
          },
          {
            h: "Discussions",
            d: "Open-ended questions, design proposals, and feedback threads.",
            href: "https://github.com/axiomchronicles/ellips/discussions",
          },
          {
            h: "Issue tracker",
            d: "Bugs and concrete feature requests with reproduction steps.",
            href: "https://github.com/axiomchronicles/ellips/issues",
          },
          {
            h: "Contributing guide",
            d: "Build, test, and review conventions — read before opening a PR.",
            href: "https://github.com/axiomchronicles/ellips/blob/main/docs/developer-guide/contributing.md",
          },
        ].map((c) => (
          <a
            key={c.h}
            href={c.href}
            target="_blank"
            rel="noreferrer"
            className="block rounded-lg border border-hairline p-5 bg-surface hover:border-ink transition"
          >
            <div className="text-ink font-semibold text-[15px]">{c.h} →</div>
            <p className="text-body text-[13.5px] mt-2 leading-relaxed">{c.d}</p>
          </a>
        ))}
      </div>

      <h2>Conduct</h2>
      <p>
        ELIPS contributors are expected to be patient, direct, and technical. Disagree about
        engineering; assume good faith about people. Maintainers reserve the right to moderate
        threads that drift from that standard.
      </p>

      <h2>Acknowledgements</h2>
      <p>
        The project leans on the public work of the HNSW authors, the broader vector-search
        community, and the FAISS team — see the ADRs for specific influences. Contributors are
        credited in the git history and in the relevant <Link to="/changelog">changelog</Link>{" "}
        entries.
      </p>
    </StandalonePage>
  );
}
