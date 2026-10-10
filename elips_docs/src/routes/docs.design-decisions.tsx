import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";

export const Route = createFileRoute("/docs/design-decisions")({
  head: () => ({
    meta: [
      { title: "Design decisions — ELIPS Docs" },
      {
        name: "description",
        content: "Architectural decision records distilled into the choices that shape ELIPS.",
      },
      { property: "og:title", content: "Design decisions — ELIPS" },
      { property: "og:description", content: "The architectural decisions behind ELIPS." },
      { property: "og:url", content: "/docs/design-decisions" },
    ],
    links: [{ rel: "canonical", href: "/docs/design-decisions" }],
  }),
  component: Page,
});

const ADRS = [
  [
    "ADR-0001",
    "C++23 core",
    "ELIPS targets C++23 to use std::span, structured bindings, designated initialisers, and concepts. The Core Guidelines are non-negotiable.",
  ],
  [
    "ADR-0002",
    "Embedded only",
    "No server, no daemon. A database is a directory, a process opens it, and a file lock coordinates writers.",
  ],
  [
    "ADR-0003",
    "HNSW as the primary index",
    "Hierarchical Navigable Small World offers the best recall / latency trade-off at the embedded scale ELIPS targets.",
  ],
  [
    "ADR-0004",
    "WAL format",
    "Record-framed with CRC32C. Recovery truncates at the first invalid record. Logical, not physical.",
  ],
  [
    "ADR-0005",
    "Isolation level",
    "Single-writer, multi-reader. Atomic batched transactions buffer in memory and commit at once.",
  ],
  [
    "ADR-0006",
    "PyBind11 bindings",
    "Lightweight, zero-overhead bindings that mirror the C++ surface 1:1 plus a typed wrapper for ergonomics.",
  ],
  [
    "ADR-0007",
    "Snapshot + WAL",
    "The snapshot is the consistent baseline; the WAL is the log of mutations since. Together they make checkpoint atomic.",
  ],
  [
    "ADR-0008",
    "File locking",
    "flock-based advisory locks. RAII-bound. No background coordination.",
  ],
  [
    "ADR-0009",
    "Dynamic schema",
    "Payloads are typed variants, not a fixed schema. Filters operate on the same type system.",
  ],
  [
    "ADR-0010",
    "EQL",
    "A small expression-oriented language with a lexer, parser, AST, and executor. EQL output is the same Filter / QueryPlan the SDK produces.",
  ],
];

const GPU_ADRS = [
  [
    "GPU-001",
    "Multi-backend strategy",
    "Backends live behind a GpuBackend abstraction, selected once at startup by BackendSelector. Metal is the implemented path; CUDA and HIP/ROCm are experimental; CPU is the universal fallback.",
  ],
  [
    "GPU-002",
    "Kernels, not external graph libs",
    "ELIPS ships its own distance and top-k kernels rather than depending on vendor graph libraries. Metal implements distance/top-k; richer GPU index kernels are future work.",
  ],
  [
    "GPU-003",
    "GPU build, CPU serve",
    "Index build can use the GPU even when serving stays on CPU. The transfer manager handles the handoff.",
  ],
  [
    "GPU-004",
    "Memory management",
    "A pool allocator keeps allocations stable across queries; the memory manager owns the lifetime.",
  ],
  [
    "GPU-005",
    "Query batching",
    "QueryBatcher coalesces concurrent single-vector queries into one launch to amortise kernel overhead.",
  ],
  [
    "GPU-006",
    "Apple unified memory",
    "On Apple silicon ELIPS skips host↔device copies entirely — the Metal backend's main advantage.",
  ],
  [
    "GPU-007",
    "Precision strategy",
    "FP32 by default; mixed precision is opt-in per index, never silent.",
  ],
  [
    "GPU-008",
    "Fallback chain",
    "Backend → device → CPU. A named-but-absent backend raises BackendUnavailableError unless fallback is requested.",
  ],
];

function Page() {
  return (
    <DocsShell eyebrow="Concepts" title="Design decisions">
      <p className="text-[18px] text-ink">
        ELIPS keeps an ADR log under <code>docs/adr/</code>. The cards below distil each record into
        the choice it made. They are listed roughly in the order the runtime was built.
      </p>

      <h2>Core ADRs</h2>
      <div className="not-prose mt-4 grid sm:grid-cols-2 gap-3">
        {ADRS.map(([id, title, body]) => {
          const slug = id.toLowerCase().replace("adr-", "ADR-");
          const file = `${slug}-${title
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-|-$/g, "")}.md`;
          return (
            <a
              key={id}
              href={`https://github.com/axiomchronicles/elips/blob/main/docs/adr/${file}`}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg border border-hairline p-5 bg-surface hover:border-ink transition block"
            >
              <div className="eyebrow text-primary mb-1">{id} ↗</div>
              <div className="text-ink font-semibold text-[15px] mb-1.5">{title}</div>
              <p className="text-body text-[13.5px] leading-relaxed">{body}</p>
            </a>
          );
        })}
      </div>

      <h2>GPU ADRs</h2>
      <p className="text-body">
        GPU support is Metal-first: the Metal backend is the implemented, implemented path (Beta, validated on Apple M4) on Apple
        silicon, CUDA and HIP/ROCm are experimental and off by default, and CPU is always available.
        SYCL and Vulkan backends were removed. See the{" "}
        <a href="/docs/gpu/overview" className="text-primary underline">GPU engine overview</a> for
        the maturity table.
      </p>
      <div className="not-prose mt-4 grid sm:grid-cols-2 gap-3">
        {GPU_ADRS.map(([id, title, body]) => {
          const file = `ADR-${id}-${title
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-|-$/g, "")}.md`;
          return (
            <a
              key={id}
              href={`https://github.com/axiomchronicles/elips/blob/main/docs/adr/${file}`}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg border border-hairline p-5 bg-surface hover:border-ink transition block"
            >
              <div className="eyebrow text-primary mb-1">{id} ↗</div>
              <div className="text-ink font-semibold text-[15px] mb-1.5">{title}</div>
              <p className="text-body text-[13.5px] leading-relaxed">{body}</p>
            </a>
          );
        })}
      </div>

      <h2>Reading the ADRs in the repo</h2>
      <p>
        The originals live under <code>docs/adr/</code> on GitHub and use a consistent template —
        context, decision, status, consequences. They are the canonical source whenever this page
        summarises something.
      </p>
    </DocsShell>
  );
}
