import { createFileRoute } from "@tanstack/react-router";
import { StandalonePage } from "../components/Page";

export const Route = createFileRoute("/contact")({
  head: () => ({
    meta: [
      { title: "Contact — ELIPS" },
      {
        name: "description",
        content:
          "How to reach the ELIPS maintainers for security disclosure, partnership, or support.",
      },
      { property: "og:title", content: "Contact — ELIPS" },
      { property: "og:description", content: "Reach the ELIPS maintainers." },
      { property: "og:url", content: "/contact" },
    ],
    links: [{ rel: "canonical", href: "/contact" }],
  }),
  component: Page,
});

function Page() {
  return (
    <StandalonePage
      eyebrow="Support"
      title="Contact"
      lede="Pick the channel that matches your message. Most things are best handled in the open on GitHub."
    >
      <div className="not-prose grid sm:grid-cols-3 gap-4 my-4">
        {[
          {
            h: "Bugs",
            d: "Open an issue with reproduction.",
            a: "GitHub issues →",
            href: "https://github.com/axiomchronicles/ellips/issues",
          },
          {
            h: "Design discussion",
            d: "Use Discussions for proposals.",
            a: "GitHub discussions →",
            href: "https://github.com/axiomchronicles/ellips/discussions",
          },
          {
            h: "Security",
            d: "Coordinate privately before public disclosure.",
            a: "Email maintainers",
            href: "mailto:security@elips.invalid",
          },
        ].map((c) => (
          <a
            key={c.h}
            href={c.href}
            target="_blank"
            rel="noreferrer"
            className="block rounded-lg border border-hairline p-5 bg-surface hover:border-ink transition"
          >
            <div className="eyebrow mb-2">{c.h}</div>
            <p className="text-body text-[14px] leading-relaxed">{c.d}</p>
            <div className="text-ink text-[14px] mt-3">{c.a}</div>
          </a>
        ))}
      </div>

      <h2>Responsible disclosure</h2>
      <p>
        If you believe you have found a security-relevant defect, please coordinate privately with
        the maintainers before public disclosure. Include reproduction steps and the commit hash you
        observed it on.
      </p>
    </StandalonePage>
  );
}
