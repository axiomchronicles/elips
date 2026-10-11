# Project History

---

## Origins: Built from First Principles

ELIPS was conceived in 2024 as an answer to a straightforward question: what
would a vector database look like if it were built like SQLite? — embedded,
in-process, requiring zero infrastructure.

The core was implemented from first principles in C++23. No third-party vector
search libraries (FAISS, hnswlib, nmslib) are used for the core indexing. Every
subsystem — the vector type, the distance kernels, the HNSW graph, the WAL, the
query language, the Python bindings — was written for ELIPS. The only external
dependencies are the C++23 standard library, GoogleTest (test-only), PyBind11
(Python bindings), and optional GPU backend libraries (cuVS, MPS, oneMKL).

Per [ADR-0001](../adr/ADR-0001-cpp23-core.md), the decision to use C++23 was
driven by the need for predictable, allocation-controlled performance for SIMD
distance kernels and graph traversal, plus first-class embeddability in both C++
and Python processes with no language runtime beyond the standard library.

The project name — **E**mbedded **L**ocal **I**ndex & **P**ersistence
**S**ystem — captures the four pillars: it is embedded (in-process), local (disk
on the host), indexed (HNSW for fast search), and persistent (WAL + snapshot).

---

## Motivation

The vector database landscape in 2024 was dominated by client-server
architectures:

- **Pinecone:** Cloud-only, closed-source, API-key access.
- **Milvus:** Server + etcd + object storage. Powerful but operationally heavy.
- **Qdrant:** Server process with REST/gRPC API. Requires deployment management.
- **Weaviate:** Server process with GraphQL API. Requires deployment management.
- **Chroma:** Embedded option available, but Python/Rust stack with different
  trade-offs.
- **FAISS:** In-process C++ library, but no persistence, no transactions, no
  query language.

For developers building CLIs, desktop applications, research notebooks, CI
pipelines, edge devices, or single-process services, deploying and managing a
separate database process is disproportionate. The friction of "spin up a vector
database" discourages adoption of semantic search in smaller projects.

ELIPS targets this gap: **vector search as a library, not a service.** If SQLite
replaced client-server RDBMS for embedded use cases, ELIPS aims to do the same
for vector search.

---

## v1.0 Scope

Version 1.0 shipped a vertically complete prototype: every layer from the domain
to the SDK is implemented and functional. The scope was deliberately bounded to
deliver a working, useful system without over-engineering for scale that most
initial users don't need.

### What Shipped

| Layer | Component | Implementation |
|-------|-----------|---------------|
| **Domain** | `Vector` | Owned float32 vector, L2 normalization, NaN/Inf validation |
| **Domain** | `RecordID` | UUIDv7, time-ordered byte layout, string serialization, FNV-1a hashing |
| **Domain** | `Record`, `Payload`, `SearchResult` | Composed value types with dynamic metadata |
| **Domain** | `ElipsError` hierarchy | 6 concrete exception types with inheriting constructors |
| **Vector Engine** | `distance()` | Ordering-normalized cosine/euclidean/dot_product with ARM NEON SIMD + scalar fallback |
| **Vector Engine** | `requires_normalization()` | Cosine auto-normalization on ingest/query |
| **Vector Engine** | Runtime dispatch | `KernelFn` function pointers with compile-time NEON selection |
| **Index Engine** | `IndexPort` | Abstract interface for pluggable indexes |
| **Index Engine** | `HierarchicalGraphIndex` | From-scratch HNSW: probabilistic level assignment, beam search, diversity heuristic, soft tombstones |
| **Index Engine** | `ExactIndex` | Brute-force linear scan for ground truth and small collections |
| **Index Engine** | `IndexFactory` | `make_index()` composition root |
| **Metadata** | `Filter` | Predicate tree: comparisons, set membership, substring match, AND/OR/NOT |
| **Metadata** | Fluent builder | Chained `.field().gte().field().equals()` API |
| **Query Engine** | `EQLLexer` | Tokenizer for EQL source, line comments, string/number/punctuation tokens |
| **Query Engine** | `EQLParser` | Recursive-descent parser, produces typed AST (`std::variant<...>`) |
| **Query Engine** | `QueryExecutor` | Dispatches AST statement variants to database operations |
| **Storage** | `WAL` | Record-based write-ahead log with CRC32C framing, append/flush semantics |
| **Storage** | `Serialization` | Binary primitives: native-endian scalar I/O, length-prefixed strings, typed payload codec, CRC32C (Castagnoli) |
| **Storage** | Snapshot | Atomic full-state serialization via temp-file + rename |
| **Storage** | Recovery | `open()`: lock → IDENTITY → snapshot load → WAL replay → live WAL |
| **Kernel** | `LockManager` | RAII advisory `flock(LOCK_EX \| LOCK_NB)` on `<dir>/LOCK` |
| **SDK** | `ElipsInstance` | Top-level handle: vault registry, lifecycle, GPU info |
| **SDK** | `Vault` | Named partition: place, seek, scan, fetch, erase, info |
| **SDK** | `Transaction` | Atomic batched: enqueue, validate, commit, auto-rollback |
| **SDK** | `Config` | Fluent builder: dimension, metric, index, graph_params, durability, GPU |
| **CLI** | `elips` | Subcommands: info, vaults, stats, verify, query, checkpoint, export, import, bench |
| **Bindings** | Python (PyBind11) | Full binding of all SDK types, `py.typed` stubs, context manager, EQL |
| **Tests** | 18 test files | Unit (per-component), integration (cross-component), recovery (crash scenarios), concurrency (locking) |
| **Benchmarks** | `elips_bench` | Configurable vector count and dimension, HNSW + exact benchmark |

### GPU Subsystem (Interfaces Defined)

The GPU engine's five port interfaces (`GpuPort`, `GpuMemoryPort`,
`GpuKernelPort`, `GpuStreamPort`, `GpuIndexPort`), the device manager,
selector, memory manager, dynamic batcher, all index types, and backend
stubs (CUDA, HIP, Metal) are defined. The interfaces are
complete; backend implementations are in progress.

---

## Deferred Features and v1.0 Hooks

Every deferred feature has a deliberate hook in v1.0 that makes it additive
rather than breaking:

| Future Capability | v1.0 Hook | Rationale |
|-------------------|-----------|-----------|
| **Per-segment indexes + compaction** | `IndexPort`; snapshot/WAL per vault | Compaction writes new segments; old ones are retired. Same index interface. |
| **Full MVCC version chains / Snapshot Isolation** | Single-writer model + txn buffer | The `Transaction` already buffers ops. Version chains add a per-record version number and a reader snapshot timestamp. |
| **Quantized indexes (PQ/OPQ/SQ), DiskANN** | `IndexPort` + `make_index` factory | New index implementations plug into the same abstract interface. Config selects via `IndexType` enum. |
| **AVX2 / AVX-512 kernels** | Function-pointer dispatch in `Metrics::Dispatch` | Add `dot_avx2`, `sql2_avx512` functions; select via CPUID in `Dispatch()` constructor. No API change. |
| **Columnar metadata, attribute B-trees, inverted/bloom** | `Filter` over `Payload` today | The `Filter::matches()` call stays the same. Internally, the predicate tree can consult columnar indexes instead of scanning payloads. |
| **Multi-reader shared locks** | `LockManager` seam | `flock(LOCK_SH)` instead of `LOCK_EX` for readers. The `LockManager` constructor already takes a path. |
| **Multi-node replication / sharding** | WAL is a logical, streamable log | Each WAL entry is self-describing (magic, op, vault, id, vector, payload, CRC). A replication stream is a WAL tail reader. |
| **Cloud object-storage adapters (S3/GCS/Azure)** | `StoragePort` adapter pattern | Replace filesystem I/O with object store PUT/GET behind a storage port. Same snapshot/WAL format. |
| **Numpy zero-copy ingestion** | `place()` takes `Vector` (owns data) | Add `place_span(std::span<const float>)` that directly indexes without copying. |
| **Async/streaming C++ APIs** | `seek()` returns `std::vector<SearchResult>` synchronously | Add `seek_async()` returning `std::future<std::vector<SearchResult>>`. Compatible with dynamic batcher. |
| **Cross-platform snapshot encoding** | Centralized `Serialization` helper | Replace native-endian `put<>`/`get<>` with little-endian encoding. Same `crc32c()` checksum. |
| **GPU CAGRA construction** | `GpuIndexPort::build_from_batch()` | Interface defined; cuVS implementation in progress. |
| **GPU IVF-PQ quantization** | `GpuQuantizationPipeline` | Pipeline defined; product quantization kernels in progress. |

---

## Known v1.0 Limitations

These are acknowledged limitations of the current implementation, documented for
transparency:

1. **Snapshot serialization uses native byte order.** Moving a database between
   big-endian and little-endian machines (e.g., aarch64_be to x86_64) produces
   incompatible snapshots. Single-machine use is unaffected.

2. **Checkpoints rewrite the entire snapshot (O(N)).** Every `checkpoint()` call
   serializes all vaults from scratch. At 10⁶ records this is acceptable
   (typically < 1 second for 768-dimensional vectors). Per-segment incremental
   persistence is deferred.

3. **Filtered ANN search post-filters an over-fetched candidate set.**
   `Vault::seek()` retrieves `top * 20` candidates from the index, then applies
   the filter and truncates. For highly selective filters (< 1% of records
   match), more candidates may be needed. Adaptive `ef` expansion is a deferred
   optimizer refinement.

4. **Single-writer at a time.** The `LockManager` acquires an exclusive lock.
   Concurrent reads from other processes work because they open the database
   read-only (no WAL), but only one writer can hold the database open at once.

5. **Indexes rebuilt on every `open()`.** Vectors are stored in the snapshot, but
   index internals (graph edges, link structures) are not serialized. On
   `open()`, vectors are re-inserted into fresh indexes. This is deterministic
   but adds startup cost proportional to dataset size.

---

## Architecture Decision Records

Twenty Architecture Decision Records (ADRs) document every significant technical
choice. Ten core ADRs and ten GPU ADRs.

### Core ADRs (10)

| # | Title | Decision |
|---|-------|----------|
| [ADR-0001](../adr/ADR-0001-cpp23-core.md) | C++23 as the core language | Modern C++ with zero-overhead abstractions, direct SIMD, trivial embedding |
| [ADR-0002](../adr/ADR-0002-embedded-only.md) | Embedded-only deployment (no server) | In-process only; database = directory; no daemon, port, or network code |
| [ADR-0003](../adr/ADR-0003-hnsw-primary-index.md) | HNSW as the primary ANN index | From-scratch HNSW with ExactIndex as ground-truth oracle; both behind `IndexPort` |
| [ADR-0004](../adr/ADR-0004-wal-format.md) | Record-based WAL with CRC32C | Logical record framing: magic \| op \| vault \| id \| [vector \| payload] \| CRC. Torn tail recovery via CRC validation |
| [ADR-0005](../adr/ADR-0005-isolation-level.md) | Atomic batched transactions for v1.0 | Eagerly validated ops buffered and applied atomically on commit. MVCC deferred |
| [ADR-0006](../adr/ADR-0006-pybind11.md) | PyBind11 for Python bindings | Maps C++ exceptions → Python, STL containers, `return_value_policy` for lifetime, `py.typed` stubs |
| [ADR-0007](../adr/ADR-0007-snapshot-plus-wal.md) | Snapshot + WAL persistence | IDENTITY for config, atomic snapshot (temp + rename), WAL replay on open. Segments deferred |
| [ADR-0008](../adr/ADR-0008-file-locking.md) | File advisory locking for coordination | Non-blocking `flock(LOCK_EX)` on `<dir>/LOCK`. RAII-bound to instance lifetime |
| [ADR-0009](../adr/ADR-0009-dynamic-schema.md) | Dynamic metadata schema | `map<string, variant<int64, double, bool, string>>`. No schema declaration. Filter compares by type |
| [ADR-0010](../adr/ADR-0010-eql-language.md) | EQL as a parsed language | Lexer → parser → AST → executor. Filter tree reused across SDK and language. |

### GPU ADRs (10)

| # | Title | Decision |
|---|-------|----------|
| [ADR-GPU-001](../adr/ADR-GPU-001-multi-backend-strategy.md) | Multi-backend abstraction strategy | Single `GpuPort` interface; all backends implement it. Backend headers never included in domain code |
| [ADR-GPU-002](../adr/ADR-GPU-002-cuvs-vs-custom-kernels.md) | cuVS vs custom CUDA kernels | Both: cuVS primary when available, custom hand-written kernels as fallback. cuBLAS for brute-force |
| [ADR-GPU-003](../adr/ADR-GPU-003-gpu-build-cpu-serve.md) | GPU build / CPU serve mode | GPU accelerates index construction (8−12×); CPU serves queries. CAGRA → HNSW export |
| [ADR-GPU-004](../adr/ADR-GPU-004-memory-management.md) | Memory management strategy | Best-fit pool allocator (`GpuMemoryManager`), pinned host memory, unified memory for Apple Silicon |
| [ADR-GPU-005](../adr/ADR-GPU-005-dynamic-batching.md) | Dynamic batching strategy | Push-model `DynamicBatcher` with configurable time window (500µs) and max batch size (256) |
| [ADR-GPU-006](../adr/ADR-GPU-006-apple-unified-memory.md) | Unified memory on Apple Silicon | `MTLResourceStorageModeShared` for zero-copy; `UnifiedBuffer` abstraction; `std::memcpy` instead of blit |
| [ADR-GPU-007](../adr/ADR-GPU-007-precision-strategy.md) | Precision selection strategy | Auto-select FP16/FP32 based on hardware. Mixed-precision pipeline: FP32 storage → FP16 GPU → FP32 reranking |
| [ADR-GPU-008](../adr/ADR-GPU-008-fallback-chain.md) | Fallback chain design | 4 levels: GPU index → GPU brute-force → CPU HNSW → CPU exact. Never return error to user |

---

## Roadmap

See [Roadmap](../roadmap.md) for the full deferred-capabilities table with v1.0
hooks.

In summary, the path forward:

### Short term (v1.1)

- GPU backend implementations (Metal first, then CUDA, HIP — CUDA/HIP experimental, not hardware-validated)
- AVX2/AVX-512 kernels via CPUID dispatch
- `ef`-adaptive filtered search
- Numpy zero-copy ingestion for Python SDK
- Serialized index format (avoid rebuild on open)
- Little-endian canonical snapshot encoding

### Medium term (v1.2+)

- Per-segment indexes with background compaction
- Full MVCC version chains with Snapshot Isolation
- Quantized indexes: Product Quantization (PQ), Optimized PQ (OPQ), Scalar Quantization (SQ)
- Columnar metadata storage with attribute indexes (B-tree, inverted, bloom)
- Multi-reader shared locks
- Async/streaming C++ APIs

### Long term (v2.0)

- DiskANN for billion-scale disk-resident search
- Multi-node replication via WAL streaming
- Cloud object-storage adapters (S3, GCS, Azure Blob)
- Sharding with consistent hashing
- GPU-native serving at query time (CAGRA online search)

---

## Development Timeline

| Period | Milestone |
|--------|-----------|
| 2024 Q3 | Project inception. Core ADRs written (0001–0010). |
| 2024 Q4 | Domain model, metrics (NEON + scalar), ExactIndex, HNSW initial implementation. |
| 2025 Q1 | Storage (WAL, snapshot, recovery), LockManager, Filter engine. |
| 2025 Q2 | EQL lexer/parser/AST/executor, CLI, Python bindings (PyBind11). |
| 2025 Q3 | Transactions, benchmark suite, test suite (18 test files). |
| 2025 Q4 | GPU ADRs written (GPU-001 through GPU-008). GPU ports defined. |
| 2026 Q1 | GPU engine: DeviceManager, MemoryManager, DynamicBatcher, all index types. |
| 2026 Q2 | GPU backend stubs (CUDA, HIP, Metal). Public interfaces complete. |

---

## Acknowledgments

ELIPS draws inspiration from:

- **SQLite** — for the embedded, in-process, zero-infrastructure philosophy.
- **FAISS** (Facebook AI Research) — for pioneering GPU-accelerated vector
  search and establishing the IVF-PQ paradigm.
- **hnswlib** (Yury Malkov et al.) — for demonstrating that HNSW can be
  implemented efficiently in a single-header C++ library.
- **Chroma** — for validating the "vector database as a library" market.
- **The C++ Core Guidelines** (Bjarne Stroustrup, Herb Sutter) — for providing a
  coherent philosophy for modern C++ that the entire ELIPS codebase follows.
- **PyBind11** — for making C++/Python interop feel seamless.