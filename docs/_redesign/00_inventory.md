# 00 — Inventory (true status) — Phase 1 synthesis, 2026-10-10

Consolidated from `_phase1_cpp.md` (1A), `_phase1_python.md` (1B), `_phase1_gpu.md` (1C),
`_phase1_docs.md` (1D). Those files carry the per-symbol `file:line` detail; this is the
true-status overview + the decisions it forces. Baseline build/test: see `05_test_matrix.md`.

## Headline (this reframes the mission)

1. **The C++ core is essentially complete, not immature.** No stub/TODO bodies in scope.
   HNSW, IVF, PQ/OPQ/SQ8, EQL, WAL/recovery, metadata filters, concurrency are all real and
   mostly tested. The brief's "immature engine" premise is **wrong for the CPU core**.
2. **The immaturity is concentrated in four places:** (a) GPU backends, (b) API naming/vocabulary,
   (c) docs/website truth, (d) specific test-coverage gaps. That is where Phases 2/5/7/9 must aim.
3. **Vocabulary exists in THREE inconsistent layers** (the real naming problem — see below).
4. **Only Metal is a real GPU backend, and only partially.** CUDA/HIP/SYCL/Vulkan are `#error` stubs.
5. **Docs/website actively contradict shipped code** (e.g. site says "there is no add/search/get/delete";
   code ships exactly those). 10 GPU ADRs describe unimplemented backends as done.
6. **Version drift:** Python reports `1.0.0`; CMake project is `1.1.0`.

## The three vocabulary layers (critical for Phase 5)

| Layer | Container | Insert | Query | Delete | Source |
|---|---|---|---|---|---|
| C++ core (1A) | `Vault` | `place` / `place_many` / `place_document` | `seek` / `seek_text` / `seek_hybrid` | `erase` | real verbs; **no** arena/ingest/probe/meta/top |
| Python modern (1B) | `Collection` + `Engine` via `connect()` | `add` / `ingest` / `write` | `search` / `probe` / `probe_text` | `delete` / `discard` | Chroma-ish surface **already exists**, delegates to native `Vault` |
| Python native (1B) | `Vault` via `open()`→`Database` | `place*` | `seek*` | `erase` | mirrors C++ |
| Docs + README + website (1D) | `Arena`/`Vault`/`Database` | `ingest`/`place` | `probe`/`seek` | `erase` | **teaches the OLD surface**; worst drift |

**Consequence for Phase 5:** the Chroma-style target (`Collection`, `add/search/get/delete`) is
**partially shipped already** in Python modern — Phase 5 is reconcile-and-complete, not greenfield.
Gaps vs target: `Client`/`PersistentClient` do **not** exist; FAISS-style `Index`/`index_factory`
public API not exposed in Python; `meta`→`metadatas`, `top=`→`n_results/k` not unified.

## C++ core status (1A) — detail in `_phase1_cpp.md`

| Engine | Status | Notes |
|---|---|---|
| domain / Config / facade | complete | `std::expected` used; the pre-C++23 polyfill is dead code |
| index_engine (ExactIndex, HierarchicalGraphIndex=HNSW, IndexFactory) | complete, tested | HNSW recall@10 ≥0.90 vs oracle |
| quant_engine (PQ/OPQ/SQ8, KMeans) | complete; component-tested | **quantized read path through indexes + snapshot/transfer: no C++ test** |
| vector_engine (Metrics) | complete | NEON implemented (comment says "SIMD later" — stale) |
| query_engine (EQL lexer/parser/executor) | complete | `==` lexed but parser-rejected; `QueryStrategy` wider than dispatch (`text_probe`/`hybrid_fusion` labels only) |
| metadata (Filter, MetadataIndex) | complete | **latent bug:** int64 stringified via `(double)` 6-sig-fig → large ints (≥~1e7) can bucket-collide; saved only by re-filter, untested |
| storage (WAL, Serialization, FileSync) | complete, recovery-tested | `WAL::append_insert_quantized`/`reset` untested in C++ |
| kernel (LockManager), text_engine, cli, benchmarks | complete | `place_many`/`seek_hybrid` have only Python-level coverage |

Deliberately unimplemented: `QuantParams::refine_fraction` (cites nonexistent ADR-0011).

## Python wrapper status (1B) — detail in `_phase1_python.md`

- Layering: `_core`(.so) → `_native` seam → {`native`, deprecated `core`/`modern`} → high-level `elips`.
- No fake natives; every facade wraps a real `_core` handle. `tests/python` = **239 passed / 5 skipped**.
- **Dead bindings:** `IndexSnapshot/IvfSnapshot/PqSnapshot/IndexSnapshotKind` built in C++, never returned by API.
- **Docstring mismatches:** `Engine` claims transactions it lacks; `elips.query` cites nonexistent `Engine.query`.
- `elips.query/search/storage/quantization` not attribute-accessible from bare `import elips`; `exceptions.py` orphaned.
- Untested in Python: GPU `Accelerator` facade; by-name aliases `add/add_many/search_vector/delete/collection_names`.

## GPU backend status (1C) — detail in `_phase1_gpu.md`

| Backend | Status | Reality |
|---|---|---|
| Metal | **partial-real** | GPU `compute_distances_batch` real; **top-k on CPU** (`std::partial_sort`); `synchronize`/`is_idle` no-ops; real 111-LOC `.metal` file is **dead code** (not in CMake), `.mm` uses a different inline kernel |
| CUDA | **stub** | `.cpp` is `#ifdef…#error`; CMake names nonexistent `CudaBackend.cu`; `.cu` kernel commented out |
| HIP | **stub** | `#error` body; CMake names nonexistent `HipBackend.hip` |
| SYCL | **stub** | `#error` body (Phase 2: DELETE) |
| Vulkan | **stub** | `#error` body; `.comp` shaders real-ish but never built, one has a shadowing compile bug (Phase 2: DELETE) |

- `GpuKernelPort`/`GpuStreamPort`: **zero implementers** (orphan interfaces).
- **No unit test runs real GPU code** — all inject `RecordingBackend` (CPU emulator). Integration/parity tests `GTEST_SKIP` without a device, so real Metal only runs on Apple hardware and is otherwise unexercised in CI.

## Docs vs code (1D) — detail in `_phase1_docs.md`

- Website `/docs/python-sdk` says "There is no add, ingest, search, get, or delete" — contradicts `collection.py:939-958`.
- 10 Accepted GPU ADRs + `gpu-engine.md` present CUDA/HIP/SYCL/Vulkan (cuVS, CAGRA, rocBLAS, SPIR-V, speedups) as implemented — false.
- Signature drift: website `connect(gpu=True)`, `engine.gpu_info()`, `ElipsInstance`, `SearchResult`, `.ge()`, `:name` EQL bindings, `SEARCH FROM … HYBRID_WEIGHT` — none match code (`Database`, `Result`, `.gte()`, grammar).
- `open()` documented with three conflicting signatures / invented kwargs; wrong `index_type`/`durability`/`dimension` defaults.
- Undocumented-but-shipped: `Collection`/`Engine`/`add`/`compress`/`health`/`CollectionHealth`/`merge`, CLI `quantize`, `QuantParams`.
- Roadmap stale (shared locks, ef-adaptive search listed as planned but shipped).

## Latent build breaks found (not touched — fix in Phase 2/8)
- CUDA/HIP CMake reference nonexistent `.cu`/`.hip` source filenames (actual files are `.cpp`).
- A referenced `src/gpu_engine/tests` directory is missing.
- Linker "ignoring duplicate libraries" (elips_core/elips_gpu linked twice).
