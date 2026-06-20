import { useEffect, useState, type ReactNode } from "react";

type Lang = "python" | "cpp" | "bash" | "eql" | "json" | "text";

const PY_KW = /\b(from|import|as|def|class|return|if|else|elif|for|in|while|with|try|except|finally|raise|pass|None|True|False|and|or|not|is|lambda|yield|await|async|print)\b/g;
const CPP_KW = /\b(auto|const|constexpr|class|struct|namespace|public|private|protected|return|if|else|for|while|do|switch|case|break|continue|template|typename|using|true|false|nullptr|new|delete|virtual|override|final|static|enum|inline|noexcept|try|catch|throw|operator)\b/g;
const BASH_KW = /\b(cmake|ctest|cd|export|python3?|pip|bun|brew|apt|yum|sudo|elips|git|make|ninja)\b/g;
const EQL_KW = /\b(seek|fetch|place|erase|scan|in|from|nearest|top|threshold|where|rank_by|project|yield|and|or|not|contains|true|false|vector|data|id|limit|offset)\b/gi;

function escape(s: string) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * Highlight without re-matching previously inserted markup. We tokenize into
 * sentinels first (so later passes can't see span attributes like `class=`),
 * escape, then substitute the sentinels back with real <span> tags.
 */
function highlight(src: string, lang: Lang): string {
  const tokens: { cls: string; text: string }[] = [];
  const mark = (cls: string) => (m: string) => {
    const i = tokens.length;
    tokens.push({ cls, text: m });
    // Pad index with Z so `\b\d+\b` (numbers) doesn't re-tokenize the id.
    return `\uE000Z${i}Z\uE001`;
  };
  let s = src;

  if (lang === "cpp") {
    s = s.replace(/\/\/[^\n]*/g, mark("tok-com"));
    s = s.replace(/\/\*[\s\S]*?\*\//g, mark("tok-com"));
    s = s.replace(/(^|\n)(\s*#[a-z]+[^\n]*)/g, (_, p, d) => p + mark("tok-pp")(d));
  }
  if (lang === "python" || lang === "bash" || lang === "eql") {
    s = s.replace(/(^|\s)(#[^\n]*)/g, (_, p, c) => p + mark("tok-com")(c));
  }
  s = s.replace(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g, mark("tok-str"));
  s = s.replace(/\b\d+\.?\d*\b/g, mark("tok-num"));
  const kw = lang === "python" ? PY_KW : lang === "cpp" ? CPP_KW : lang === "bash" ? BASH_KW : lang === "eql" ? EQL_KW : null;
  if (kw) s = s.replace(kw, mark("tok-kw"));
  if (lang === "cpp") {
    s = s.replace(/\b(std|elips|Config|Vector|Filter|Vault|ElipsInstance|Metric|AccessMode|DocumentAttachment|ChunkInfo|EmbeddingLineage|QueryPlan|IndexPort|GpuPort|HierarchicalGraphIndex|ExactIndex|LockManager|WAL)\b/g, mark("tok-ty"));
  }
  if (lang === "python") {
    s = s.replace(/\b(elips|Filter|Config|RecordInput|Row|Hit|DocumentAttachment|ChunkInfo|EmbeddingLineage)\b/g, mark("tok-ty"));
    s = s.replace(/\b([a-z_][a-z0-9_]*)(?=\()/gi, mark("tok-fn"));
  }

  s = escape(s);
  s = s.replace(/\uE000Z(\d+)Z\uE001/g, (_m, i) => {
    const t = tokens[Number(i)];
    if (!t) return "";
    return `<span class="${t.cls}">${escape(t.text)}</span>`;
  });
  return s;
}

export function CodeBlock({
  children,
  lang = "python",
  filename,
}: {
  children: string;
  lang?: Lang;
  filename?: string;
}) {
  const [copied, setCopied] = useState(false);
  const html = highlight(children.replace(/\n+$/, ""), lang);
  return (
    <div className="code-surface my-6">
      <div className="code-head">
        <span>{filename ?? lang}</span>
        <button
          onClick={() => {
            navigator.clipboard?.writeText(children);
            setCopied(true);
            setTimeout(() => setCopied(false), 1400);
          }}
          className="tracking-normal normal-case text-[11px] opacity-70 hover:opacity-100 transition"
        >
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <pre>
        <code dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  );
}

export function Inline({ children }: { children: ReactNode }) {
  return <code className="font-mono text-[0.88em] px-1.5 py-0.5 rounded bg-canvas-soft border border-hairline">{children}</code>;
}

export function Mermaid({ chart, caption }: { chart: string; caption?: string }) {
  const [svg, setSvg] = useState<string>("");
  const [theme, setTheme] = useState<"light" | "dark">("light");
  useEffect(() => {
    const obs = new MutationObserver(() => {
      setTheme(document.documentElement.classList.contains("dark") ? "dark" : "light");
    });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    setTheme(document.documentElement.classList.contains("dark") ? "dark" : "light");
    return () => obs.disconnect();
  }, []);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const mermaid = (await import("mermaid")).default;
      const isDark = theme === "dark";
      // Editorial palette — paper for light, warm ink for dark — with a
      // hand-drawn look so diagrams feel illustrated, not generic.
      const bg = isDark ? "#2a2924" : "#fbfaf6";
      const stroke = isDark ? "#e8e3d4" : "#26251e";
      const text = isDark ? "#f4efe2" : "#1a1915";
      const accent = "#f54e00";
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",
        look: "classic",
        theme: "base",
        fontFamily: "'Caveat', 'Patrick Hand', 'Kalam', 'Inter', system-ui, sans-serif",
        themeVariables: {
          background: bg,
          primaryColor: bg,
          primaryTextColor: text,
          primaryBorderColor: stroke,
          secondaryColor: isDark ? "#34322c" : "#f3eee0",
          secondaryTextColor: text,
          secondaryBorderColor: stroke,
          tertiaryColor: isDark ? "#3a3830" : "#efe8d6",
          tertiaryTextColor: text,
          tertiaryBorderColor: stroke,
          lineColor: stroke,
          textColor: text,
          mainBkg: bg,
          nodeBorder: stroke,
          clusterBkg: isDark ? "#23221d" : "#f6f1e2",
          clusterBorder: stroke,
          edgeLabelBackground: bg,
          labelTextColor: text,
          actorBkg: bg,
          actorBorder: stroke,
          actorTextColor: text,
          signalColor: stroke,
          signalTextColor: text,
          noteBkgColor: isDark ? "#3a3830" : "#fff6d6",
          noteTextColor: text,
          noteBorderColor: stroke,
          activationBkgColor: accent,
          fontSize: "16px",
        },
        flowchart: { curve: "basis", padding: 24, htmlLabels: true, useMaxWidth: true },
        sequence: { actorMargin: 60, messageMargin: 40, mirrorActors: false, useMaxWidth: true },
      });
      const id = "m" + Math.random().toString(36).slice(2, 9);
      try {
        const { svg } = await mermaid.render(id, chart);
        if (!cancelled) setSvg(svg);
      } catch (e) {
        if (!cancelled) setSvg(`<pre style="color:#cf2d56">${String(e)}</pre>`);
      }
    })();
    return () => { cancelled = true; };
  }, [chart, theme]);
  return (
    <figure className="my-10 sketch-figure">
      <div className="sketch-frame">
        <span className="sketch-corner sketch-corner-tl" aria-hidden />
        <span className="sketch-corner sketch-corner-tr" aria-hidden />
        <span className="sketch-corner sketch-corner-bl" aria-hidden />
        <span className="sketch-corner sketch-corner-br" aria-hidden />
        <div className="mermaid-host" dangerouslySetInnerHTML={{ __html: svg || '<div style="height:160px"/>' }} />
      </div>
      {caption && (
        <figcaption className="sketch-caption">
          <span className="sketch-caption-mark" aria-hidden>✎</span>
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
