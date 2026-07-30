// Canonical doc index used for nav + search.
export type DocEntry = {
  path: string;
  title: string;
  group: string;
  description: string;
  keywords?: string[];
};

export const docs: DocEntry[] = [
  // ── Start ────────────────────────────────────────────────────────────────
  {
    path: "/docs",
    title: "Getting started",
    group: "Start",
    description:
      "Install ELIPS, open your first database, and run a text-first search in five minutes.",
  },
  {
    path: "/docs/getting-started",
    title: "Quick start",
    group: "Start",
    description:
      "In-memory hello world, persistent DB, embedders, filtering, hybrid search — complete runnable examples.",
    keywords: ["tutorial", "quickstart", "hello world", "first steps"],
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

  // ── Concepts ─────────────────────────────────────────────────────────────
  {
    path: "/docs/project-history",
    title: "Project history",
    group: "Concepts",
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
      "HNSW, exact search, metadata acceleration, hybrid fusion, GPU index family, and the query planner.",
    keywords: ["hnsw", "ivf", "pq", "brute force", "cosine", "euclidean", "dot product"],
  },
  {
    path: "/docs/design-decisions",
    title: "Design decisions",
    group: "Concepts",
    description: "Architectural decision records distilled into the choices that shape ELIPS.",
  },

  // ── Python API ───────────────────────────────────────────────────────────
  {
    path: "/docs/python-sdk",
    title: "Python SDK overview",
    group: "Python API",
    description:
      "Two-surface architecture: low-level bindings (Database/Vault/Config) and the modern wrapper (Engine/Arena).",
    keywords: ["python", "sdk", "bindings", "pybind11"],
  },
  {
    path: "/docs/python/connect",
    title: "elips.connect()",
    group: "Python API",
    description:
      "Modern initialization factory: path, dimension, metric, index_type, embedder, durability, and access modes.",
    keywords: ["connect", "initialize", "factory", "in-memory"],
  },
  {
    path: "/docs/python/engine",
    title: "Engine",
    group: "Python API",
    description:
      "High-level Engine wrapper: arena(), checkpoint(), compact(), vacuum(), vault_names(), pending_writes(), close().",
    keywords: ["engine", "database handle", "context manager"],
  },
  {
    path: "/docs/python/arena",
    title: "Arena",
    group: "Python API",
    description:
      "Typed collection wrapper: write(), write_many(), ingest(), probe(), probe_text(), probe_hybrid(), pull(), sweep(), discard(), health().",
    keywords: ["arena", "collection", "vault wrapper", "write", "search", "probe"],
  },
  {
    path: "/docs/python/models",
    title: "Data models",
    group: "Python API",
    description:
      "RecordInput, Row, Hit, WalRecord, ArenaHealth, DocumentAttachment, ChunkInfo, EmbeddingLineage.",
    keywords: ["record input", "hit", "row", "data model", "typed"],
  },
  {
    path: "/docs/python/database",
    title: "Database (low-level)",
    group: "Python API",
    description:
      "Low-level Database class: vault(), list_vaults(), begin_transaction(), query(), checkpoint(), compact(), close().",
    keywords: ["database", "low-level", "elipsinstance"],
  },
  {
    path: "/docs/python/vault",
    title: "Vault (low-level)",
    group: "Python API",
    description:
      "Low-level Vault: place(), place_document(), place_many(), seek(), seek_text(), seek_hybrid(), explain_seek(), scan(), fetch(), erase().",
    keywords: ["vault", "place", "seek", "search", "collection", "low-level"],
  },
  {
    path: "/docs/python/config",
    title: "Config & GraphParams",
    group: "Python API",
    description:
      "Config fluent builder, GraphParams HNSW tuning, LocalEmbedderConfig, durability levels, access modes.",
    keywords: ["config", "configuration", "graph params", "hnsw", "durability"],
  },
  {
    path: "/docs/python/transaction",
    title: "Transactions",
    group: "Python API",
    description:
      "Atomic batched writes: begin_transaction(), TransactionVault.place/erase, commit(), rollback(), context manager.",
    keywords: ["transaction", "atomic", "commit", "rollback"],
  },
  {
    path: "/docs/python/filtering",
    title: "Filtering",
    group: "Python API",
    description:
      "Filter fluent builder, static factories, boolean combinators, metadata acceleration, seek integration.",
    keywords: ["filter", "metadata", "where", "comparator", "in_set", "contains"],
  },
  {
    path: "/docs/python/durability",
    title: "Durability & WAL",
    group: "Python API",
    description:
      "Durability levels, WAL guarantees, atomic commit with undo log, access modes, and LockConflict.",
  },
  {
    path: "/docs/python/wal",
    title: "Recovery & introspection",
    group: "Python API",
    description:
      "WAL replay, crash forensics, EQL AST, index snapshots, embedder introspection, and vector utilities.",
  },
  {
    path: "/docs/python/maintenance",
    title: "Index maintenance",
    group: "Python API",
    description:
      "vacuum(), pending_removals, rebuild_index, records snapshot, sealed, read-only toggle, and database lifecycle.",
  },
  {
    path: "/docs/python/errors",
    title: "Exceptions & errors",
    group: "Python API",
    description:
      "ElipsError hierarchy: LockConflictError, ValidationError, TransactionError, DiskError, and GpuError handling.",
    keywords: ["error", "exception", "lockconflict", "validationerror", "try except"],
  },
  {
    path: "/docs/python/utilities",
    title: "Vector utilities",
    group: "Python API",
    description:
      "Cosine similarity, L2 distance, normalization, NumPy interop, and batch embedder helper functions.",
    keywords: ["utilities", "vector math", "numpy", "normalize", "l2"],
  },
  {
    path: "/docs/python/eql",
    title: "Python EQL",
    group: "Python API",
    description:
      "Execute EQL statements in Python via db.query() and engine.query() with parameterized vector bindings.",
    keywords: ["eql", "query", "sql", "bindings", "search"],
  },
  {
    path: "/docs/python/gpu",
    title: "GPU (Python)",
    group: "Python API",
    description:
      "Python GPU API: device discovery, GpuDevice, Accelerator, compute_distances, top_k, memory pools, profiling.",
    keywords: ["gpu", "cuda", "hip", "metal", "accelerator", "device"],
  },

  // ── C++ API ──────────────────────────────────────────────────────────────
  {
    path: "/docs/cpp-sdk",
    title: "C++ SDK overview",
    group: "C++ API",
    description:
      "C++23 surface — Config, ElipsInstance, Vault, transactions, embedders, query plans, locking, and GPU.",
    keywords: ["c++", "cpp", "sdk", "c++23"],
  },
  {
    path: "/docs/cpp/overview",
    title: "C++ Engine Architecture",
    group: "C++ API",
    description:
      "High-performance C++23 native vector engine: headers, RAII memory, thread-safety, and CMake target linking.",
    keywords: ["cpp overview", "architecture", "headers", "cmake", "raii"],
  },
  {
    path: "/docs/cpp/elips-instance",
    title: "ElipsInstance",
    group: "C++ API",
    description:
      "Top-level database handle: open(), vault(), checkpoints, WAL management, compaction, and process locking.",
    keywords: ["elipsinstance", "open", "checkpoint", "vacuum", "close"],
  },
  {
    path: "/docs/cpp/vault",
    title: "Vault",
    group: "C++ API",
    description:
      "Vector & payload collection: place(), seek(), seek_hybrid(), explain_seek(), scan(), fetch(), erase(), vacuum().",
    keywords: ["vault", "place", "seek", "hybrid", "explain_seek", "vacuum"],
  },
  {
    path: "/docs/cpp/domain",
    title: "Domain Types & Record Model",
    group: "C++ API",
    description:
      "RecordID, Vector, Payload, Record, SearchResult, attachments, and C++ exception hierarchies.",
    keywords: ["domain", "record", "searchresult", "payload", "exceptions"],
  },
  {
    path: "/docs/cpp/config",
    title: "Config & GraphParams",
    group: "C++ API",
    description:
      "Config struct, GraphParams HNSW parameters, Metric enums, Durability policies, and performance tuning recipes.",
    keywords: ["config", "graphparams", "metric", "durability", "tuning"],
  },
  {
    path: "/docs/cpp/transaction",
    title: "C++ Transactions",
    group: "C++ API",
    description:
      "Atomic multi-vault write transactions: PendingOp buffer, two-phase commit, undo log, and exception safety.",
    keywords: ["transaction", "commit", "rollback", "atomic", "undo"],
  },

  // ── GPU ──────────────────────────────────────────────────────────────────
  {
    path: "/docs/gpu-engine",
    title: "GPU Engine Overview",
    group: "GPU",
    description:
      "Hardware-accelerated vector search — CUDA, HIP, Metal backends, memory pools, dynamic batching, and GPU index family.",
    keywords: ["gpu", "cuda", "hip", "metal", "acceleration"],
  },
  {
    path: "/docs/gpu/overview",
    title: "GPU Architecture & Pipelines",
    group: "GPU",
    description:
      "GpuPort abstractions, dynamic batcher, memory allocators (PinnedBuffer/UnifiedBuffer), and search pipelines.",
    keywords: ["gpu architecture", "gpuport", "batcher", "pinnedbuffer", "pipeline"],
  },

  // ── Reference ────────────────────────────────────────────────────────────
  {
    path: "/docs/api",
    title: "API reference",
    group: "Reference",
    description:
      "Complete Python and C++ API surface — Database, Vault, Config, Filter, query plan, and EQL.",
  },
  {
    path: "/docs/eql",
    title: "EQL language",
    group: "Reference",
    description: "ELIPS Query Language — grammar, SEARCH/FETCH/SCAN/INSERT/DELETE, filters, projections, and bindings.",
    keywords: ["eql", "query language", "grammar", "sql-like"],
  },
  {
    path: "/docs/cli",
    title: "CLI",
    group: "Reference",
    description:
      "The elips command — info, vaults, stats, verify, query, checkpoint, import, export, bench.",
    keywords: ["cli", "command line", "elips bench"],
  },
  {
    path: "/docs/benchmarks",
    title: "Benchmarks & Performance",
    group: "Reference",
    description:
      "Benchmark suite: QPS throughput, p50/p95/p99 latency, recall trade-offs, HNSW tuning, and GPU scaling.",
    keywords: ["benchmarks", "performance", "qps", "latency", "recall", "tuning"],
  },

  // ── Internals ────────────────────────────────────────────────────────────
  {
    path: "/docs/internals/lock-manager",
    title: "Lock manager",
    group: "Internals",
    description:
      "Single-writer / multi-reader through a POSIX advisory file lock, RAII-bound to LockManager.",
    keywords: ["lock", "flock", "concurrency", "process lock"],
  },
  {
    path: "/docs/internals/transaction-engine",
    title: "Transaction engine",
    group: "Internals",
    description:
      "Atomic batched writes — eager validation, PendingOp buffer, commit, rollback, and the Python context-manager binding.",
  },

  // ── Practice ─────────────────────────────────────────────────────────────
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
    description:
      "Task-shaped walkthroughs: in-memory quick start, persistent DB, batch ingestion, custom embedder, filtering, hybrid search, GPU, transactions, production durability.",
    keywords: ["guide", "how to", "tutorial", "walkthrough"],
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
    keywords: ["advanced", "production", "shared reader", "planner"],
  },
  {
    path: "/docs/performance",
    title: "Performance & benchmarks",
    group: "Practice",
    description:
      "Tuning HNSW, choosing exact vs. graph, GPU batching, durability impact, and benchmarking.",
    keywords: ["performance", "tuning", "benchmark", "throughput", "recall"],
  },
  {
    path: "/docs/security",
    title: "Security & privacy",
    group: "Practice",
    description:
      "Local-only data, advisory locking, WAL guarantees, and operational considerations.",
  },

  // ── Project ──────────────────────────────────────────────────────────────
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
  { to: "/docs/gpu-engine", label: "GPU" },
  { to: "/docs/eql", label: "EQL" },
  { to: "/docs/guides", label: "Guides" },
  { to: "/docs/roadmap", label: "Roadmap" },
];

export const groups = [
  "Start",
  "Concepts",
  "Python API",
  "C++ API",
  "GPU",
  "Reference",
  "Internals",
  "Practice",
  "Project",
] as const;
