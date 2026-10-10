# 03 — Naming Map (ADR) — Phase 5, 2026-10-10

**Status: ACCEPTED.** Deep rename through the core (user decision, ADR-R2 in `04_architecture.md`).
This is the single source of truth for the rename; the exhaustive per-symbol tables live in
`_naming_cpp.md` (344 C++ rows), `_naming_python.md`, `_naming_other.md` (EQL/CMake/tests/files).
Renames are applied mechanically in batches, keeping ctest + pytest green after each batch, with a
deprecation shim mapping old public names → new (DeprecationWarning, removed next minor).

## Reality corrections (verified in code — reality wins, rule #7)
- There is **no `class Database`** in C++. The engine type is `ElipsInstance` (`elips.hpp:290`).
  Decision: `ElipsInstance`→`Database`; add `Client`/`PersistentClient` wrappers on top. `src/Database.cpp` keeps its filename.
- Real C++ verbs are `place*`/`seek*`/`erase`/`fetch`/`scan` — there is no `ingest`/`probe`/`meta`/`top=` in C++ (those are Python/docs-layer).

## DO NOT RENAME (safety — would break behavior or on-disk compat)
- `nprobe`, `probe_lists`, `probe_all_devices` — FAISS-standard / device-probe terms, **not** the search-`probe`. Leave.
- On-disk `"vault_<n>_<epoch>.segment"` prefix and `elips.manifest`/snapshot magic (`Database.cpp:1687`) — persisted format. Keep; a format rename needs a migration (defer to Phase 6/8, not this rename).
- `Filter::and_/or_/not_` already dodge keyword clashes — leave.
- Layers with no public-vocabulary drift stay unchanged: domain, quant_engine, storage, text_engine, vector_engine, kernel.

## Collision rule
Where a Chroma name is a C++ keyword/reserved, the C++ method takes a safe synonym and Python exposes the Chroma name:
- `delete` → C++ **`remove`**, Python **`delete`**, EQL keyword **`delete`** (DSL, safe).

## Resolved open collisions
1. One public `Collection` (Python user-facing + C++ core class). The native pybind handle is internal.
2. `fetch`→`get` everywhere (C++ `get`, Python `get`, EQL `get`); unifies with Chroma `get`.
3. kwarg `documents` (plural, Chroma) for the text payload; the single-attachment type stays `DocumentAttachment`.
4. `list_collections` returns Collection objects (Chroma-compatible); add `list_collection_names()` for names.
5. `n_results` is primary; `k` and `limit` are accepted aliases.
6. EQL `top`→`k`; `data`→`metadata`. `scan`/`nearest`/`from`/`where`/`yield`/`vector`/`rank_by`/`project` kept.
7. `GraphParams`→`HnswParams`; `IndexType::graph/exact`→`IndexType::hnsw/flat` (keep old enumerators as deprecated aliases).

## Canonical map by surface (headline rows; full tables in `_naming_*.md`)

### C++ classes / types  (`_naming_cpp.md`)
| Current | New |
|---|---|
| `Vault` | `Collection` |
| `ElipsInstance` | `Database` (+ new `Client`, `PersistentClient`) |
| `IndexPort` | `Index` (abstract base) |
| `ExactIndex` | `IndexFlat` |
| `HierarchicalGraphIndex` | `IndexHNSW` |
| *(new)* | `IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer` |
| `IndexFactory` / `make_index` | `index_factory` |
| `MetadataIndex` | `MetadataFilterIndex` |
| `GpuPort` | `GpuBackend` |
| `GpuIndexPort` | `GpuIndex` |
| `GpuSelector` | `BackendSelector` |
| `DynamicBatcher` | `QueryBatcher` |
| `GraphParams` | `HnswParams` |
| `GpuKernelPort`, `GpuStreamPort` | **delete** (zero implementers) |

### C++ methods (on Collection/Index)
`place`→`add`, `place_many`→`add_many`, `place_document`→`add_document`, `seek`→`search`,
`seek_text`→`search_text`, `seek_hybrid`→`hybrid_search`, `explain_seek`→`explain_query`,
`erase`→`remove`, `fetch`→`get`, `scan`→`scan` (kept). Index contract ADD: `train`, `add_with_ids`
(was `insert`), `range_search`, `reconstruct`, `reconstruct_n`, `remove`, `reset`, `is_trained`,
`ntotal` (was `size`), `d`, `metric_type`.

### Python  (`_naming_python.md`)
Entry: ADD `Client()`, `PersistentClient(path=)`; `connect`/`open`/`Engine`→deprecated shims→`Client`.
ADD `create_collection`/`get_collection`/`get_or_create_collection`/`delete_collection`/`list_collections`.
Collection verbs → Chroma nine: `add/upsert/get/update/delete/query/count/peek/modify`
(old `write/ingest/merge/probe*/pull/sweep/discard/seek*` → shims). kwargs: `top=`→`n_results`(alias `k`),
`meta=`→`metadatas`, `keys=`→`ids`, `vector(s)=`→`embeddings`, `text=`→`documents`,
`include_vectors`→`include=[...]`; ADD `where`, `where_document`, `query_embeddings/query_texts`, `limit/offset`.
Expose FAISS layer: `IndexFlat/IndexIVFFlat/IndexIVFPQ/IndexHNSW`, `index_factory`, `read_index/write_index`,
`index_cpu_to_gpu/index_gpu_to_cpu`, `get_num_gpus`, `StandardGpuResources`. Retire dead `IndexSnapshot*` (no shim).

### EQL  (`_naming_other.md`)
`place`→`add`, `seek`→`search`, `fetch`→`get`, `erase`→`delete`, `data`→`metadata`, `top`→`k`;
`scan`/`in`/`from`/`where`/`nearest`/`vector`/`yield` kept. AST `InsertStatement`→`AddStatement`, `.vault`→`.collection`.
Lexer is keyword-agnostic (no change). Delete the fabricated uppercase `SEARCH FROM…SEEK :q TOP` website dialect.

### CMake  (`_naming_other.md`)
Options: `ELIPS_GPU_METAL/CUDA/HIP`→`ELIPS_ENABLE_METAL/CUDA/ROCM`; `ELIPS_BUILD_PYTHON`→`ELIPS_ENABLE_PYTHON`;
`ELIPS_BUILD_TESTS`→`BUILD_TESTING`; add `ELIPS_ENABLE_GPU` (umbrella), `ELIPS_ENABLE_C_API`, `BUILD_SHARED_LIBS`,
`ELIPS_OPT_LEVEL` (generic/neon/avx2/avx512). Targets kept recognizable (`elips_core`, `elips_gpu_metal`, …).

### Files
`MetadataIndex.*`→`MetadataFilterIndex.*`, `GpuSelector.*`→`BackendSelector.*`, `DynamicBatcher.*`→`QueryBatcher.*`,
`GpuPort.hpp`→`GpuBackend.hpp`, `IndexPort.hpp`→`Index.hpp`. `Database.cpp` keeps its name (now matches `Database`).

### Tests
GTest + pytest names carrying old vocab renamed (place/seek/erase/vault → add/search/delete/collection);
`metadata_index_test.cpp`→`metadata_filter_index_test.cpp`, `DatabaseGpuLifecycleProbe.cpp`→`ClientGpuLifecycleTest.cpp`.
Do NOT rename device-`probe` test names.

### Errors
17 `vault`→`collection` user-facing strings in `src/Database.cpp`; `*Error` suffix alignment
(`DimensionMismatch`→`DimensionMismatchError`, …); ADD `NotTrainedError`/`CollectionNotFoundError`/`BackendUnavailableError`;
wire the orphaned `exceptions.py`.

## Deprecation shim policy
Every renamed PUBLIC name keeps an old-name alias for one minor version:
- Python: thin wrappers / `__getattr__` emitting `DeprecationWarning`; documented in `MIGRATION.md`.
- C++: `[[deprecated("use X")]]` inline alias functions / `using` aliases in public headers.
- EQL: old verbs accepted by the parser with a one-time warning (no uppercase dialect).
- CMake: old option names honored via `if(DEFINED ELIPS_GPU_METAL) set(ELIPS_ENABLE_METAL …)` with a `message(DEPRECATION …)`.
Final grep for old terms must be empty **except** the shim definitions and `MIGRATION.md`.

## Application plan (mechanical, verified per batch)
1. C++ core types+methods (headers first, then src) → build + ctest green.
2. File renames (git mv) + include fixups → build green.
3. EQL verbs + AST → ctest (eql tests) green.
4. pybind layer + Python facades + kwargs → pytest green.
5. CMake option aliases + target/test renames → fresh configure+build+ctest green.
6. Error strings + typed exceptions.
7. Deprecation shim + `MIGRATION.md`; final old-term grep (empty except shim+MIGRATION).
Docs (Phase 9) consume this map; not part of the code-rename batches.
