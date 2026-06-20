import { createFileRoute } from "@tanstack/react-router";
import { StandalonePage } from "../components/Page";

export const Route = createFileRoute("/privacy")({
  head: () => ({
    meta: [
      { title: "Privacy — ELIPS" },
      {
        name: "description",
        content:
          "What the ELIPS documentation site stores, and what the ELIPS runtime does (and does not) do with your data.",
      },
      { property: "og:title", content: "Privacy — ELIPS" },
      { property: "og:description", content: "What we store and what we do not." },
      { property: "og:url", content: "/privacy" },
    ],
    links: [{ rel: "canonical", href: "/privacy" }],
  }),
  component: Page,
});

function Page() {
  return (
    <StandalonePage
      eyebrow="Legal"
      title="Privacy policy"
      lede="The shortest version: this site stores a theme preference and a consent flag in your browser. The runtime never sends your data anywhere."
    >
      <h2>The documentation site</h2>
      <ul>
        <li>
          <strong>Local storage:</strong> two keys — <code>elips-theme</code> and{" "}
          <code>elips-cookie-consent</code>. Both live in your browser only.
        </li>
        <li>
          <strong>Analytics:</strong> none. No third-party trackers, no pixels, no fingerprinting.
        </li>
        <li>
          <strong>Server logs:</strong> standard request logs may be retained by the hosting
          provider; no personal identifiers are correlated.
        </li>
      </ul>

      <h2>The runtime</h2>
      <p>
        ELIPS is embedded. It does not open a network socket on your behalf, does not call out to
        remote services, and does not phone home. Every record you ingest stays in the database
        directory you pass on open.
      </p>

      <h2>Rights</h2>
      <p>
        You may clear the documentation site's local storage at any time through your browser. There
        are no accounts to delete and no personal data held server-side.
      </p>

      <h2>Contact</h2>
      <p>
        Questions about this policy can be sent through the channels on the{" "}
        <a href="/contact">contact page</a>.
      </p>
    </StandalonePage>
  );
}
