import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  Link,
  createRootRouteWithContext,
  useRouter,
  HeadContent,
  Scripts,
} from "@tanstack/react-router";
import { type ReactNode } from "react";

import appCss from "../styles.css?url";
import { TopNav, Footer, CookieConsent } from "../components/Chrome";
import { ShortcutsModal } from "../components/Shortcuts";

function NotFoundComponent() {
  return (
    <div className="min-h-[70vh] flex items-center justify-center px-6">
      <div className="max-w-md text-center">
        <div className="eyebrow mb-3">404 · not found</div>
        <h1 className="display-lg text-ink">This page hasn't been written.</h1>
        <p className="mt-3 text-body text-[15px]">
          The URL doesn't match a known route. The docs map lives below.
        </p>
        <div className="mt-6 flex gap-2 justify-center">
          <Link to="/" className="btn btn-ghost">
            Home
          </Link>
          <Link to="/docs" className="btn btn-primary">
            Read the docs
          </Link>
        </div>
      </div>
    </div>
  );
}

function ErrorComponent({ error, reset }: { error: Error; reset: () => void }) {
  console.error(error);
  const router = useRouter();
  return (
    <div className="min-h-[70vh] flex items-center justify-center px-6">
      <div className="max-w-md text-center">
        <div className="eyebrow mb-3">Something broke</div>
        <h1 className="display-lg text-ink">This page didn't load.</h1>
        <p className="mt-3 text-body">An unexpected error occurred while rendering.</p>
        <div className="mt-6 flex gap-2 justify-center">
          <button
            onClick={() => {
              router.invalidate();
              reset();
            }}
            className="btn btn-primary"
          >
            Try again
          </button>
          <a href="/" className="btn btn-ghost">
            Go home
          </a>
        </div>
      </div>
    </div>
  );
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "ELIPS — Embedded Local Index & Persistence System" },
      {
        name: "description",
        content:
          "ELIPS is an in-process vector and document retrieval engine built in C++23, with native Python bindings, ANN indexes, hybrid retrieval, and segmented persistence.",
      },
      { name: "author", content: "ELIPS contributors" },
      { name: "theme-color", content: "#f7f7f4" },
      { property: "og:site_name", content: "ELIPS Docs" },
      { property: "og:type", content: "website" },
      { property: "og:title", content: "ELIPS — Embedded Local Index & Persistence System" },
      {
        property: "og:description",
        content:
          "Local-first ANN, hybrid retrieval, document lineage, and segmented persistence — embedded in your process.",
      },
      { name: "twitter:card", content: "summary_large_image" },
      { name: "twitter:title", content: "ELIPS — Embedded vector & document engine" },
      {
        name: "twitter:description",
        content: "An embedded vector and document retrieval engine in C++23. Python and C++ SDKs.",
      },
    ],
    links: [
      { rel: "stylesheet", href: appCss },
      { rel: "preconnect", href: "https://fonts.googleapis.com" },
      { rel: "preconnect", href: "https://fonts.gstatic.com", crossOrigin: "anonymous" },
      {
        rel: "stylesheet",
        href: "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Caveat:wght@400;500;600;700&family=Kalam:wght@300;400;700&display=swap",
      },
      {
        rel: "icon",
        href: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%23f54e00'/%3E%3Ctext x='50%25' y='58%25' text-anchor='middle' font-family='Inter,sans-serif' font-size='18' font-weight='700' fill='white'%3Ee%3C/text%3E%3C/svg%3E",
      },
    ],
    scripts: [
      {
        type: "application/ld+json",
        children: JSON.stringify({
          "@context": "https://schema.org",
          "@type": "SoftwareApplication",
          name: "ELIPS",
          description:
            "Embedded Local Index & Persistence System — an in-process vector and document retrieval engine.",
          applicationCategory: "DeveloperApplication",
          operatingSystem: "macOS, Linux, Windows",
          offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
        }),
      },
    ],
  }),
  shellComponent: RootShell,
  component: RootComponent,
  notFoundComponent: NotFoundComponent,
  errorComponent: ErrorComponent,
});

function RootShell({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body>
        {children}
        <Scripts />
      </body>
    </html>
  );
}

function RootComponent() {
  const { queryClient } = Route.useRouteContext();
  return (
    <QueryClientProvider client={queryClient}>
      <TopNav />
      <Outlet />
      <Footer />
      <CookieConsent />
      <ShortcutsModal />
    </QueryClientProvider>
  );
}
