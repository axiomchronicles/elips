import { Link, useRouterState } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Lightbulb } from "lucide-react";
import { docs, siteNav } from "../lib/content";
import { AskAI } from "./AskAI";

/* ---------------- Version switcher ---------------- */
const VERSIONS = [
  { id: "1.0", label: "1.0 (stable)", current: true },
  { id: "0.9", label: "0.9", current: false },
  { id: "0.8", label: "0.8", current: false },
  { id: "next", label: "Next (preview)", current: false },
];
export function VersionSwitcher() {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState("1.0");
  useEffect(() => {
    const saved = localStorage.getItem("elips-docs-version");
    if (saved && VERSIONS.some((v) => v.id === saved)) setActive(saved);
  }, []);
  const v = VERSIONS.find((x) => x.id === active) ?? VERSIONS[0];
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="h-9 px-2.5 inline-flex items-center gap-1.5 rounded-md border border-hairline-strong text-[12px] text-body hover:text-ink hover:border-ink transition"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="font-mono text-[11px] text-primary">v{v.id}</span>
        <span aria-hidden>▾</span>
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-11 z-40 w-56 bg-surface border border-hairline rounded-lg p-1.5">
            <div className="eyebrow px-2.5 py-1.5">Documentation version</div>
            {VERSIONS.map((vv) => (
              <button
                key={vv.id}
                onClick={() => {
                  setActive(vv.id);
                  localStorage.setItem("elips-docs-version", vv.id);
                  setOpen(false);
                }}
                className={`w-full text-left px-2.5 py-2 rounded-md text-[13px] flex items-center justify-between transition ${
                  active === vv.id
                    ? "bg-canvas-soft text-ink"
                    : "text-body hover:bg-canvas-soft hover:text-ink"
                }`}
              >
                <span>{vv.label}</span>
                {vv.current && (
                  <span className="text-[10px] uppercase tracking-wider text-primary">current</span>
                )}
              </button>
            ))}
            <div className="px-2.5 py-2 mt-1 border-t border-hairline-soft text-[11px] text-muted leading-snug">
              Archived versions stay live at the same URL prefixed with the version id.
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/* ---------------- Theme ---------------- */
export function ThemeToggle() {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const saved = localStorage.getItem("elips-theme");
    const prefers = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
    const isDark = saved ? saved === "dark" : !!prefers;
    setDark(isDark);
    document.documentElement.classList.toggle("dark", isDark);
  }, []);
  function toggle() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("elips-theme", next ? "dark" : "light");
  }
  return (
    <button
      onClick={toggle}
      aria-label="Toggle theme"
      className="h-9 w-9 inline-flex items-center justify-center rounded-md border border-hairline-strong hover:border-ink transition text-ink"
    >
      <span className="text-[15px]">{dark ? "☾" : "☀"}</span>
    </button>
  );
}

/* ---------------- Search ---------------- */
export function SearchTrigger() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const query = q.trim().toLowerCase();
  const results = !query
    ? docs.slice(0, 8).map((d) => ({ d, score: 0, snippet: d.description }))
    : docs
        .map((d) => {
          const title = d.title.toLowerCase();
          const desc = d.description.toLowerCase();
          const group = d.group.toLowerCase();
          const words = query.split(/\s+/).filter(Boolean);
          let score = 0;
          for (const w of words) {
            if (title === w) score += 100;
            if (title.startsWith(w)) score += 40;
            if (title.includes(w)) score += 25;
            if (group.includes(w)) score += 8;
            if (desc.includes(w)) score += 4;
            if (d.path.includes(w)) score += 6;
          }
          // Build a tiny preview snippet around the first hit.
          const first = words.find((w) => desc.includes(w));
          let snippet = d.description;
          if (first) {
            const i = desc.indexOf(first);
            const start = Math.max(0, i - 40);
            snippet = (start > 0 ? "…" : "") + d.description.slice(start, start + 140) + "…";
          }
          return { d, score, snippet };
        })
        .filter((r) => r.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 12);
  const groupSet = Array.from(new Set(results.map((r) => r.d.group)));
  const [groupFilter, setGroupFilter] = useState<string | null>(null);
  const visible = groupFilter ? results.filter((r) => r.d.group === groupFilter) : results;
  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="h-9 px-3 inline-flex items-center gap-3 rounded-md border border-hairline-strong text-muted hover:border-ink hover:text-ink transition text-[13px] min-w-[200px]"
      >
        <span>Search the docs</span>
        <span className="ml-auto font-mono text-[11px] border border-hairline rounded px-1.5 py-0.5">
          ⌘K
        </span>
      </button>
      {open && (
        <div
          className="fixed inset-0 z-50 bg-ink/30 backdrop-blur-sm flex items-start justify-center pt-[10vh] px-4 fade-up"
          onClick={() => {
            setOpen(false);
            setGroupFilter(null);
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-2xl bg-surface border border-hairline rounded-xl overflow-hidden"
          >
            <input
              autoFocus
              value={q}
              onChange={(e) => {
                setQ(e.target.value);
                setGroupFilter(null);
              }}
              placeholder="Search pages, APIs, concepts…"
              className="w-full px-5 py-4 bg-transparent text-ink outline-none border-b border-hairline text-[15px]"
            />
            {groupSet.length > 1 && (
              <div className="flex gap-2 px-5 py-2.5 border-b border-hairline-soft overflow-x-auto">
                <button
                  onClick={() => setGroupFilter(null)}
                  className={`text-[11px] px-2.5 py-1 rounded-full border transition ${groupFilter === null ? "bg-ink text-canvas border-ink" : "border-hairline-strong text-muted hover:text-ink"}`}
                >
                  All ({results.length})
                </button>
                {groupSet.map((g) => {
                  const n = results.filter((r) => r.d.group === g).length;
                  return (
                    <button
                      key={g}
                      onClick={() => setGroupFilter(g)}
                      className={`text-[11px] px-2.5 py-1 rounded-full border transition ${groupFilter === g ? "bg-ink text-canvas border-ink" : "border-hairline-strong text-muted hover:text-ink"}`}
                    >
                      {g} ({n})
                    </button>
                  );
                })}
              </div>
            )}
            <ul className="max-h-[55vh] overflow-y-auto">
              {visible.length === 0 && (
                <li className="px-5 py-6 text-muted text-sm">
                  No matches for "{q}".{" "}
                  <Link
                    to="/chat"
                    search={{ q } as never}
                    onClick={() => setOpen(false)}
                    className="text-ink underline"
                  >
                    Ask AI instead →
                  </Link>
                </li>
              )}
              {visible.map((r) => (
                <li key={r.d.path}>
                  <Link
                    to={r.d.path}
                    onClick={() => {
                      setOpen(false);
                      setGroupFilter(null);
                    }}
                    className="block px-5 py-3 hover:bg-canvas-soft transition border-b border-hairline-soft last:border-b-0"
                  >
                    <div className="flex items-baseline justify-between gap-4">
                      <div className="text-ink text-[15px]">{r.d.title}</div>
                      <div className="eyebrow">{r.d.group}</div>
                    </div>
                    <div className="text-muted text-[13px] mt-0.5 line-clamp-2">{r.snippet}</div>
                    <div className="text-muted-soft text-[11px] mt-1 font-mono">{r.d.path}</div>
                  </Link>
                </li>
              ))}
              {query && visible.length > 0 && (
                <li className="border-t border-hairline">
                  <Link
                    to="/chat"
                    search={{ q } as never}
                    onClick={() => {
                      setOpen(false);
                      setGroupFilter(null);
                    }}
                    className="flex items-center gap-3 px-5 py-3 hover:bg-canvas-soft transition text-[14px] text-ink"
                  >
                    <Lightbulb className="askai-spark" size={14} aria-hidden />
                    Ask AI: <span className="text-muted">"{q}"</span>
                  </Link>
                </li>
              )}
            </ul>
          </div>
        </div>
      )}
    </>
  );
}

/* ---------------- Top Nav ---------------- */
export function TopNav() {
  return (
    <header className="sticky top-0 z-40 bg-canvas/85 backdrop-blur border-b border-hairline">
      <div className="max-w-[1200px] mx-auto h-16 px-6 flex items-center gap-8">
        <Link
          to="/"
          className="text-ink text-[18px] tracking-tight"
          style={{ letterSpacing: "-0.02em" }}
        >
          <span className="text-primary">elips</span>
          <span className="text-muted">/docs</span>
        </Link>
        <nav className="hidden md:flex items-center gap-7 ml-2">
          {siteNav.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              className="text-[14px] text-body hover:text-ink transition"
              activeProps={{ className: "text-ink" }}
            >
              {l.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-3">
          <SearchTrigger />
          <Link
            to="/chat"
            className="hidden sm:inline-flex h-9 px-3 items-center gap-1.5 rounded-md text-[13px] text-body hover:text-ink transition"
          >
            <Lightbulb size={14} aria-hidden /> Ask AI
          </Link>
          <a
            href="https://github.com/axiomchronicles/elips"
            target="_blank"
            rel="noreferrer"
            className="hidden sm:inline-flex h-9 px-3 items-center text-[13px] text-body hover:text-ink transition"
          >
            GitHub →
          </a>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

/* ---------------- Footer ---------------- */
export function Footer() {
  return (
    <footer className="hairline-t mt-32">
      <div className="max-w-[1200px] mx-auto px-6 py-16 grid grid-cols-2 md:grid-cols-5 gap-10">
        <div className="col-span-2">
          <div className="text-ink text-[20px]" style={{ letterSpacing: "-0.02em" }}>
            <span className="text-primary">elips</span>
          </div>
          <p className="text-body text-[14px] mt-3 max-w-sm">
            Embedded Local Index &amp; Persistence System. An in-process vector and document
            retrieval engine, built in C++23.
          </p>
        </div>
        {[
          {
            h: "Docs",
            l: [
              ["Getting started", "/docs"],
              ["Architecture", "/docs/architecture"],
              ["API reference", "/docs/api"],
              ["EQL", "/docs/eql"],
            ],
          },
          {
            h: "Project",
            l: [
              ["Changelog", "/changelog"],
              ["Contributing", "/contributing"],
              ["Roadmap", "/docs/design-decisions"],
              ["FAQ", "/faq"],
            ],
          },
          {
            h: "Support",
            l: [
              ["Ask AI", "/chat"],
              ["Help center", "/help"],
              ["Community", "/community"],
              ["Contact", "/contact"],
            ],
          },
        ].map((col) => (
          <div key={col.h}>
            <div className="eyebrow mb-3">{col.h}</div>
            <ul className="space-y-2">
              {col.l.map(([label, href]) => (
                <li key={href}>
                  <Link to={href} className="text-[14px] text-body hover:text-ink transition">
                    {label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="hairline-t">
        <div className="max-w-[1200px] mx-auto px-6 py-6 flex flex-wrap items-center justify-between gap-3 text-[13px] text-muted">
          <div>© {new Date().getFullYear()} ELIPS contributors. Built in the open on GitHub.</div>
          <div className="flex gap-5 items-center">
            <button
              onClick={() => window.dispatchEvent(new KeyboardEvent("keydown", { key: "?" }))}
              className="hover:text-ink inline-flex items-center gap-1.5"
            >
              <span className="kbd !h-5 !min-w-[20px] !text-[10px]">?</span> Shortcuts
            </button>
            <Link to="/terms" className="hover:text-ink">
              Terms
            </Link>
            <Link to="/privacy" className="hover:text-ink">
              Privacy
            </Link>
            <Link to="/cookies" className="hover:text-ink">
              Cookies
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}

/* ---------------- Cookie consent ---------------- */
export function CookieConsent() {
  const [show, setShow] = useState(false);
  useEffect(() => {
    if (!localStorage.getItem("elips-cookie-consent")) setShow(true);
  }, []);
  if (!show) return null;
  function decide(v: "accepted" | "declined") {
    localStorage.setItem("elips-cookie-consent", v);
    setShow(false);
  }
  return (
    <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-6 md:bottom-6 z-40 max-w-md bg-surface border border-hairline-strong rounded-lg p-5 fade-up">
      <div className="eyebrow mb-1.5">Cookie notice</div>
      <p className="text-[13px] text-body leading-relaxed">
        This documentation site stores a theme preference and a single consent flag in your browser.
        No analytics, no third-party trackers.{" "}
        <Link to="/cookies" className="underline">
          Read more
        </Link>
        .
      </p>
      <div className="flex gap-2 mt-3">
        <button
          onClick={() => decide("accepted")}
          className="btn btn-primary !h-9 !px-4 text-[13px]"
        >
          Acknowledge
        </button>
        <button onClick={() => decide("declined")} className="btn btn-ghost !h-9 !px-4 text-[13px]">
          Dismiss
        </button>
      </div>
    </div>
  );
}

/* ---------------- Docs Layout ---------------- */
export function DocsShell({
  children,
  title,
  eyebrow,
  toc,
}: {
  children: React.ReactNode;
  title?: string;
  eyebrow?: string;
  toc?: { id: string; label: string }[];
}) {
  const path = useRouterState({ select: (s) => s.location.pathname });
  const grouped = docs.reduce<Record<string, typeof docs>>((acc, d) => {
    (acc[d.group] ||= []).push(d);
    return acc;
  }, {});
  return (
    <div className="max-w-[1280px] mx-auto px-6 grid grid-cols-12 gap-10 pt-10">
      <aside className="hidden lg:block col-span-3">
        <div className="sticky top-24 max-h-[calc(100vh-7rem)] overflow-y-auto pr-2">
          {Object.entries(grouped).map(([g, items]) => (
            <div key={g} className="mb-7">
              <div className="eyebrow mb-2">{g}</div>
              <ul className="space-y-1">
                {items.map((d) => {
                  const active = path === d.path;
                  return (
                    <li key={d.path}>
                      <Link
                        to={d.path}
                        className={`block py-1 text-[14px] transition ${
                          active ? "text-primary" : "text-body hover:text-ink"
                        }`}
                      >
                        {d.title}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
      </aside>

      <main className="col-span-12 lg:col-span-7 pb-24 fade-up">
        {(eyebrow || title) && (
          <header className="mb-10 pb-6 border-b border-hairline">
            {eyebrow && <div className="eyebrow mb-3">{eyebrow}</div>}
            {title && <h1 className="display-lg text-ink !leading-[1.1] !mb-0">{title}</h1>}
          </header>
        )}
        <article className="prose">{children}</article>
        <div className="hairline-t mt-16 pt-6 text-[13px] text-muted flex justify-between">
          <Link to="/docs" className="hover:text-ink">
            ← All docs
          </Link>
          <a
            href="https://github.com/axiomchronicles/elips"
            target="_blank"
            rel="noreferrer"
            className="hover:text-ink"
          >
            Edit on GitHub →
          </a>
        </div>
      </main>

      <aside className="hidden xl:block col-span-2">
        {toc && toc.length > 0 && (
          <div className="sticky top-24">
            <div className="eyebrow mb-3">On this page</div>
            <ul className="space-y-1.5">
              {toc.map((t) => (
                <li key={t.id}>
                  <a href={`#${t.id}`} className="text-[13px] text-muted hover:text-ink transition">
                    {t.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </aside>
      <AskAI context={title} />
    </div>
  );
}
