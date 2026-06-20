// Canonical doc index used for nav + search.
export type DocEntry = {
  path: string;
  title: string;
  group: string;
  description: string;
  keywords?: string[];
};

export const docs: DocEntry[] = [
  {
    path: "/docs",
    title: "Getting started",
    group: "Start",
    description:
      "Install ELIPS, open your first database, and run a text-first search in five minutes.",
  },
  {
    path: "/docs/installation",
    title: "Installation",
    group: "Start",
    description: "Build the C++ core, install the Python bindings, and verify your toolchain.",
  },
  {
    path: "/docs/configuration",
    title: "Configuration",
    group: "Start",
    description:
      "Configure dimension, metric, index type, durability, segmented storage, and the text embedder.",
  },

  {
    path: "/docs/project-history",
    title: "Project history",
    group: "Overview",
    description: "Why ELIPS exists, the SQLite-for-vectors thesis, and what v1.0 set out to prove.",
  },
  {
    path: "/docs/core-concepts",
    title: "Core concepts",
    group: "Concepts",
    description:
      "Engines, vaults, records, documents, chunks, lineage — the mental model behind ELIPS.",
  },
  {
    path: "/docs/architecture",
    title: "Architecture",
    group: "Concepts",
    description: "How ElipsInstance, Vault, the planner, and the persistence layer fit together.",
  },
  {
    path: "/docs/storage",
    title: "Storage & recovery",
    group: "Concepts",
    description:
      "On-disk layout, the WAL, segmented persistence, checkpoint, compaction, and crash recovery.",
  },
  {
    path: "/docs/algorithms",
    title: "Algorithms",
    group: "Concepts",
    description:
      "HNSW, exact search, metadata acceleration, hybrid fusion, and the GPU index family.",
  },
  {
    path: "/docs/design-decisions",
    title: "Design decisions",
    group: "Concepts",
    description: "Architectural decision records distilled into the choices that shape ELIPS.",
  },

  {
    path: "/docs/internals/lock-manager",
    title: "Lock manager",
    group: "Internals",
    description:
      "Single-writer / multi-reader through a POSIX advisory file lock, RAII-bound to LockManager.",
  },
  {
    path: "/docs/internals/transaction-engine",
    title: "Transaction engine",
    group: "Internals",
    description:
      "Atomic batched writes — eager validation, PendingOp buffer, commit, rollback, and the Python context-manager binding.",
  },
  {
    path: "/docs/gpu-engine",
    title: "GPU engine",
    group: "Internals",
    description:
      "Optional GPU acceleration — ports, CUDA/HIP/Metal backends, dynamic batching, memory pools, and the GpuIndexPort family.",
  },

  {
    path: "/docs/api",
    title: "API reference",
    group: "Reference",
    description:
      "The Python and C++ surfaces — Database, Vault, Config, Filter, query plan, and EQL.",
  },
  {
    path: "/docs/python-sdk",
    title: "Python SDK",
    group: "Reference",
    description:
      "Low-level bindings, the modern Engine/Arena wrapper, embedders, EQL, transactions, and read-only serving.",
  },
  {
    path: "/docs/cpp-sdk",
    title: "C++ SDK",
    group: "Reference",
    description:
      "C++23 surface — Config, ElipsInstance, Vault, transactions, embedders, query plans, locking, and GPU configuration.",
  },
  {
    path: "/docs/eql",
    title: "EQL",
    group: "Reference",
    description: "ELIPS Query Language — grammar, statements, filters, projections, and bindings.",
  },
  {
    path: "/docs/cli",
    title: "CLI",
    group: "Reference",
    description:
      "The elips command — info, vaults, stats, verify, query, checkpoint, import, export, bench.",
  },

  {
    path: "/docs/tutorial",
    title: "Tutorial (16 lessons)",
    group: "Practice",
    description:
      "Hand-drawn, sixteen-lesson walkthrough of the entire ELIPS engine — Python first, C++ where it matters.",
  },
  {
    path: "/docs/guides",
    title: "Guides",
    group: "Practice",
    description: "Task-shaped walkthroughs for common ELIPS workflows.",
  },
  {
    path: "/docs/examples",
    title: "Examples",
    group: "Practice",
    description: "Concrete Python and C++ snippets straight from the repository.",
  },
  {
    path: "/docs/advanced",
    title: "Advanced patterns",
    group: "Practice",
    description:
      "Custom embedders, hybrid fusion, planner introspection, and shared-reader serving.",
  },
  {
    path: "/docs/performance",
    title: "Performance",
    group: "Practice",
    description:
      "Tuning HNSW, choosing exact vs. graph, GPU batching, and benchmarking with elips bench.",
  },
  {
    path: "/docs/security",
    title: "Security & privacy",
    group: "Practice",
    description:
      "Local-only data, advisory locking, WAL guarantees, and operational considerations.",
  },

  {
    path: "/docs/roadmap",
    title: "Roadmap",
    group: "Project",
    description:
      "What v1.0 ships today and what is intentionally deferred, with the v1.0 hook that keeps each future capability additive.",
  },
  {
    path: "/changelog",
    title: "Changelog",
    group: "Project",
    description: "What shipped in each ELIPS release.",
  },
  {
    path: "/contributing",
    title: "Contributing",
    group: "Project",
    description: "Coding standards, testing matrix, and the release process.",
  },
  {
    path: "/faq",
    title: "FAQ",
    group: "Project",
    description: "Common questions about embedding, deployment, and operational behavior.",
  },
];

export const siteNav = [
  { to: "/docs", label: "Docs" },
  { to: "/docs/python-sdk", label: "Python" },
  { to: "/docs/cpp-sdk", label: "C++" },
  { to: "/docs/eql", label: "EQL" },
  { to: "/docs/roadmap", label: "Roadmap" },
  { to: "/community", label: "Community" },
];

export const groups = [
  "Start",
  "Overview",
  "Concepts",
  "Internals",
  "Reference",
  "Practice",
  "Project",
] as const;
