# 02 — Gap Analysis (ELIPS vs FAISS v1.15.1 vs Chroma) — Phase 4 synthesis, 2026-10-10

Consolidated from `_gap_4A.md` (index/search/CRUD/persistence), `_gap_4B.md` (GPU/Metal),
`_gap_4C.md` (API/build/concurrency/tests/docs). Those hold the full per-row tables with
`file:line`; this is the prioritized register. **Every gap below has an owner phase (Phase 4 gate).**
Severity: blocker / major / minor. Effort: S / M / L.

## Blockers (must-fix for the recognizable-API + validated-GPU goal)

| # | Gap | ELIPS today | Sev | Eff | Owner |
|---|---|---|---|---|---|
| B1 | Vocabulary chaos: 3 live surfaces (C++ `place`/`seek`/`Vault`; Py `add`/`search`/`Collection`; docs `arena`/`ingest`/`probe`) | inconsistent, docs deny shipped API | blocker | M | 5 |
| B2 | No `index_factory` string grammar | only fluent `Config`→`make_index` (`IndexFactory`) | blocker | M | 6 |
| B3 | No Chroma `Client`/`PersistentClient` entry points | only `connect()`/`open()` | blocker | S | 5 |
| B4 | GPU top-k on CPU (`std::partial_sort`, `MetalBackend.mm:251`) | not a real GPU index | blocker | M | 7 |
| B5 | GPU distance kernel = naive triple-loop in 1×1×1 threadgroup (`MetalBackend.mm:236`) | not performant/correct-at-scale | blocker | M | 7 |
| B6 | No test runs a real GPU kernel (all `RecordingBackend` CPU emulator) | Metal unvalidated | blocker | M | 7 |
| B7 | Docs/website actively wrong (deny `add/search`; 10 GPU ADRs overclaim) | misleads users | blocker | L | 9 (gated on 5) |

## Phase 5 (naming / public API) — mostly S
- Unify `top=`→`n_results` (alias `k`), `meta`→`metadatas` (`_gap_4C`). minor/S.
- Add `Client` (ephemeral) + `PersistentClient(path=)` wrapping existing engine (B3). blocker/S.
- Collection API: add `upsert`/`update`/`peek` names, `get`, `where`/`where_document`/`include=` (dict `$`-ops) — some are aliases of existing verbs (`place` already upsert, `Database.cpp:898`). major/M.
- Keep ELIPS descriptive index-param names; record FAISS aliases (`M`/`efConstruction`/`efSearch`, `nlist`/`nprobe`). minor/S.
- Deprecation shim old→new (`arena`/`ingest`/`probe`/`seek`/`place_document`/`erase`/`top=`/`meta`). major/M.
- Retire dead Python bindings `IndexSnapshot*`; fix `Engine`/`elips.query` docstring lies. minor/S.

## Phase 6 (core engine) — the real "maturity" work (reframed: fill gaps, don't rebuild)
- `index_factory("IVF4096,PQ16")` parser over existing index types (B2). blocker/M.
- **CPU IVF family (IVFFlat/IVFPQ/IVF-SQ) — currently GPU-only** (`Config.hpp:31` has only graph+exact). major/L. *(biggest single core gap)*
- `IndexRefine`/exact-rerank — implement `QuantParams::refine_fraction` (dead knob, cites nonexistent ADR-0011). major/M.
- Batched k-NN (`search` over N queries) + true radius `range_search`. major/M.
- Metric semantics: `euclidean` returns true √L2 not FAISS squared (`Metrics.cpp:107`) — document + optional squared mode; cosine=IP-on-normalized. major/S.
- `k>ntotal`: ELIPS clamps to min(k,n); FAISS pads `-1` — pick+document (recommend FAISS pad for parity). minor/S.
- numpy boundary: enforce float32 + C-contiguity + `(n,d)` (today `py::iterable` + per-elem cast, no zero-copy, `elips_python.cpp`). major/M → also Phase 8.
- Serialization: add magic+version header to index I/O (ELIPS snapshots already have it; align index-level). minor/S.
- Add typed errors `NotTrainedError`/`CollectionNotFoundError`/`BackendUnavailableError` (others exist). minor/S.
- Concurrency: ELIPS search is single-threaded (no OpenMP/pool anywhere) — add a threading policy + document reader/writer contract. major/M.

## Phase 7 (GPU, Metal-first) — adapt FAISS's Metal precedent (`_faiss_3I_metal.md`)
- Port `topk_threadgroup` to replace CPU top-k (B4); real tiled-GEMM distance (B5). blocker/M each.
- Wire `xcrun→.air→.metallib` build chain; revive the dead `.metal` (`metal/CMakeLists.txt:7`). blocker/M.
- Real-kernel parity tests (retire `RecordingBackend` for the Metal path) (B6). blocker/M.
- `GpuResources`/`StandardGpuResources` analog (temp buffer 2× growth, 256 MiB tile budget); drop unused `MetalAllocType`. major/M.
- Per-index GPU config (device, fp16 storage, indices 32/64-bit, interleaved, precomputed tables); wire the dead fp16 flags. major/M.
- Standalone `bf_knn`/`GpuDistanceParams` (host-or-device). major/M.
- Explicit device selection + **logged** fallback (today silent; `RequireGpu` is a dead distinction, `GpuDeviceManager.cpp:111`). major/S.
- Fix selector ranking (stub CUDA/HIP ranked above real Metal, `GpuSelector.cpp:8`). minor/S.
- Query tiling, dataset>device error, OOM→typed error (today swallowed, `GpuBruteForceIndex.cpp:50`); max-k cap; metric/dtype validation. major/M.
- GPU IVF scan kernels (`ivf_scan_list*`+`ivf_merge_lists`), IVF-SQ — later. major/L → some Phase 8.
- CUDA/HIP: implement same interfaces, compile-check only, label EXPERIMENTAL. major/L.

## Phase 8 (build / packaging / test infra)
- CMake `ELIPS_ENABLE_*` + `ELIPS_OPT_LEVEL` (generic/neon/avx2/avx512), BLAS/Accelerate selection, `install(EXPORT)` + config package, per-opt-level targets. major/M.
- Fix latent breaks: missing `src/gpu_engine/tests` subdir (ELIPS_GPU_TESTS), CUDA/HIP CMake name nonexistent `.cu`/`.hip`, duplicate-library link warning. major/S.
- Packaging: abi3 wheel, cpu/gpu split rationale, scikit-build-core vs setuptools, version drift (py 1.0.0 vs CMake 1.1.0), 3.14 classifier/CI mismatch. major/M.
- Test infra: SIFT1M/GloVe recall-QPS harness + `evaluate()` methodology; CPU-Flat oracle + three-knob parity; backend-parametrized suite; cover quantized read-path + `place_many`/`seek_hybrid`; docs-snippet runner; pytest config; sanitizer/coverage. major/L.

## Phase 9 (docs) — gated on Phase 5 renames
- Regenerate reference from source; fix the website (stop denying `add/search/get/delete`); maturity labels; remove 4-backend overclaim; delete fabricated uppercase EQL dialect; fix signature/default drift (B7). blocker/L.

## ELIPS is AHEAD of FAISS — keep, do not regress
- Durability: WAL + CRC32C + magic+versioned snapshots (FAISS has no-magic fourcc).
- Hybrid lexical+vector fusion (`seek_hybrid`) — no FAISS equivalent.
- `DynamicBatcher` — no FAISS equivalent.
- Correct unified-memory zero-copy transfer on Apple Silicon.
- Built-in local text embedder with auto-provisioning.

**Gate check:** every gap row above carries an owner phase (5/6/7/8/9). ✅
