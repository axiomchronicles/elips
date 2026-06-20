import { createFileRoute } from "@tanstack/react-router";
import { StandalonePage } from "../components/Page";

export const Route = createFileRoute("/cookies")({
  head: () => ({
    meta: [
      { title: "Cookie notice — ELIPS" },
      {
        name: "description",
        content:
          "What the ELIPS documentation site stores in your browser — and what it deliberately does not.",
      },
      { property: "og:title", content: "Cookie notice — ELIPS" },
      { property: "og:description", content: "Cookie and local-storage notice." },
      { property: "og:url", content: "/cookies" },
    ],
    links: [{ rel: "canonical", href: "/cookies" }],
  }),
  component: Page,
});

function Page() {
  return (
    <StandalonePage
      eyebrow="Legal"
      title="Cookie notice"
      lede="This site uses local storage, not cookies. Two keys, both first-party, both essential to the experience."
    >
      <h2>What we store</h2>
      <table>
        <thead>
          <tr>
            <th>Key</th>
            <th>Purpose</th>
            <th>Lifetime</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <code>elips-theme</code>
            </td>
            <td>Remembers your light/dark choice.</td>
            <td>Until you clear it.</td>
          </tr>
          <tr>
            <td>
              <code>elips-cookie-consent</code>
            </td>
            <td>Records that you have dismissed the consent banner.</td>
            <td>Until you clear it.</td>
          </tr>
        </tbody>
      </table>

      <h2>What we do not use</h2>
      <ul>
        <li>No analytics cookies.</li>
        <li>No third-party trackers.</li>
        <li>No advertising or remarketing pixels.</li>
        <li>No session cookies — there are no accounts.</li>
      </ul>

      <h2>Manage</h2>
      <p>
        You can clear the keys above through your browser's site-data settings. Dismissing the
        consent banner stores the consent flag and nothing else.
      </p>
    </StandalonePage>
  );
}
