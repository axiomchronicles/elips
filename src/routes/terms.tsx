import { createFileRoute } from "@tanstack/react-router";
import { StandalonePage } from "../components/Page";

export const Route = createFileRoute("/terms")({
  head: () => ({
    meta: [
      { title: "Terms — ELIPS" },
      { name: "description", content: "Terms governing use of the ELIPS documentation site and the ELIPS open-source software." },
      { property: "og:title", content: "Terms — ELIPS" },
      { property: "og:description", content: "Documentation and software terms." },
      { property: "og:url", content: "/terms" },
    ],
    links: [{ rel: "canonical", href: "/terms" }],
  }),
  component: Page,
});

function Page() {
  return (
    <StandalonePage eyebrow="Legal" title="Terms & conditions" lede="This page covers the documentation site itself. The ELIPS software is governed by the licence file in the repository.">
      <h2>The documentation site</h2>
      <p>
        This site is provided for informational purposes. The content is
        published as-is without warranty of any kind. We make every effort
        to keep documentation accurate against the current release, but
        the source code in the repository is the canonical reference.
      </p>

      <h2>The software</h2>
      <p>
        ELIPS is open-source software developed in public on GitHub. A
        formal licence file is not yet present in the repository — until
        one is added, treat the source code's repository terms as the
        canonical statement and consult the maintainers before
        redistribution. Nothing on this site grants additional rights
        beyond what the repository itself grants.
      </p>

      <h2>Trademarks</h2>
      <p>
        "ELIPS" and the ELIPS wordmark refer to the open-source project
        and its associated community. Other names and marks referenced
        across the documentation belong to their respective owners.
      </p>

      <h2>Changes</h2>
      <p>
        These terms may be updated to reflect changes to the project or
        the way the documentation site is operated. Material changes will
        be noted in the <a href="/changelog">changelog</a>.
      </p>
    </StandalonePage>
  );
}
