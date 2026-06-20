import { useEffect, useState } from "react";

const SHORTCUTS: { keys: string[]; label: string; group: string }[] = [
  { keys: ["⌘", "K"], label: "Open search palette", group: "Navigation" },
  { keys: ["?"], label: "Open this shortcut guide", group: "Navigation" },
  { keys: ["G", "H"], label: "Go to home", group: "Navigation" },
  { keys: ["G", "D"], label: "Go to docs", group: "Navigation" },
  { keys: ["G", "A"], label: "Go to API reference", group: "Navigation" },
  { keys: ["G", "C"], label: "Go to changelog", group: "Navigation" },
  { keys: ["G", "T"], label: "Open AI chat", group: "Navigation" },
  { keys: ["T"], label: "Toggle light / dark theme", group: "Display" },
  { keys: ["Esc"], label: "Close any open dialog", group: "Display" },
];

export function ShortcutsModal() {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    let last = "";
    let lastAt = 0;
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement)?.isContentEditable)
        return;
      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        setOpen((o) => !o);
        return;
      }
      if (e.key === "Escape") {
        setOpen(false);
        return;
      }
      if (e.key.toLowerCase() === "t" && !e.metaKey && !e.ctrlKey) {
        const root = document.documentElement;
        const next = !root.classList.contains("dark");
        root.classList.toggle("dark", next);
        localStorage.setItem("elips-theme", next ? "dark" : "light");
        return;
      }
      const now = Date.now();
      if (last === "g" && now - lastAt < 900) {
        const map: Record<string, string> = {
          h: "/",
          d: "/docs",
          a: "/docs/api",
          c: "/changelog",
          t: "/chat",
        };
        const dest = map[e.key.toLowerCase()];
        if (dest) {
          window.location.href = dest;
          last = "";
          return;
        }
      }
      last = e.key.toLowerCase();
      lastAt = now;
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!open) return null;
  const grouped = SHORTCUTS.reduce<Record<string, typeof SHORTCUTS>>((acc, s) => {
    (acc[s.group] ??= []).push(s);
    return acc;
  }, {});

  return (
    <div
      className="fixed inset-0 z-50 bg-ink/30 backdrop-blur-sm flex items-center justify-center px-4 fade-up"
      onClick={() => setOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard shortcuts"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-xl bg-surface border border-hairline rounded-2xl overflow-hidden"
      >
        <div className="px-6 py-5 border-b border-hairline flex items-center justify-between">
          <div>
            <div className="eyebrow">Keyboard</div>
            <h2 className="text-[20px] text-ink font-normal mt-1">Shortcuts</h2>
          </div>
          <button
            className="text-muted hover:text-ink text-sm"
            onClick={() => setOpen(false)}
            aria-label="Close"
          >
            Esc
          </button>
        </div>
        <div className="p-6 space-y-6">
          {Object.entries(grouped).map(([group, items]) => (
            <div key={group}>
              <div className="eyebrow mb-3">{group}</div>
              <ul className="space-y-2.5">
                {items.map((s) => (
                  <li key={s.label} className="flex items-center justify-between gap-4">
                    <span className="text-[14px] text-ink">{s.label}</span>
                    <span className="flex items-center gap-1.5">
                      {s.keys.map((k, i) => (
                        <span key={i} className="kbd">
                          {k}
                        </span>
                      ))}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
