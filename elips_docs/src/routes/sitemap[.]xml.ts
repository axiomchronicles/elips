import { createFileRoute } from "@tanstack/react-router";
import { docs } from "../lib/content";
import { lessons } from "../lib/tutorial";

const BASE_URL = process.env.VITE_SITE_URL || "https://elips.tubox.cloud";

interface SitemapEntry {
  path: string;
  changefreq?: "daily" | "weekly" | "monthly" | "yearly";
  priority?: string;
}

export const Route = createFileRoute("/sitemap.xml")({
  server: {
    handlers: {
      GET: async () => {
        const staticPages: SitemapEntry[] = [
          { path: "/", changefreq: "weekly", priority: "1.0" },
          { path: "/chat", changefreq: "weekly", priority: "0.8" },
          { path: "/faq", changefreq: "monthly", priority: "0.7" },
          { path: "/help", changefreq: "monthly", priority: "0.7" },
          { path: "/community", changefreq: "monthly", priority: "0.6" },
          { path: "/contact", changefreq: "yearly", priority: "0.5" },
          { path: "/changelog", changefreq: "weekly", priority: "0.7" },
          { path: "/contributing", changefreq: "monthly", priority: "0.6" },
          { path: "/terms", changefreq: "yearly", priority: "0.3" },
          { path: "/privacy", changefreq: "yearly", priority: "0.3" },
          { path: "/cookies", changefreq: "yearly", priority: "0.2" },
        ];

        const docsEntries: SitemapEntry[] = docs.map((d) => ({
          path: d.path,
          changefreq: "weekly",
          priority: "0.8",
        }));

        const tutorialEntries: SitemapEntry[] = lessons.map((l) => ({
          path: `/docs/tutorial/${l.slug}`,
          changefreq: "weekly",
          priority: "0.8",
        }));

        const entries = [...staticPages, ...docsEntries, ...tutorialEntries];

        const urls = entries.map((e) =>
          [
            `  <url>`,
            `    <loc>${BASE_URL}${e.path}</loc>`,
            e.changefreq ? `    <changefreq>${e.changefreq}</changefreq>` : null,
            e.priority ? `    <priority>${e.priority}</priority>` : null,
            `  </url>`,
          ]
            .filter(Boolean)
            .join("\n"),
        );

        const xml = [
          `<?xml version="1.0" encoding="UTF-8"?>`,
          `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">`,
          ...urls,
          `</urlset>`,
        ].join("\n");

        return new Response(xml, {
          headers: {
            "Content-Type": "application/xml",
            "Cache-Control": "public, max-age=3600",
          },
        });
      },
    },
  },
});
