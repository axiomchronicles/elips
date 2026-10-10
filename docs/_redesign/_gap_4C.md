# Phase 4C — Gap Analysis (API ergonomics / build / concurrency / validation / tests / docs)

Branch `redesign`. Analysis only — no code changed. Evidence from Phase-1 inventories
(`00_inventory.md`, `_phase1_cpp.md`, `_phase1_python.md`, `_phase1_docs.md`), FAISS/Chroma
studies (`_faiss_3A/3E/3F/3G/3H`), and direct re-verification of `bindings/python/`,
`CMakeLists.txt`, `.github/workflows/`, `tests/` (cited `file:line`).

Columns: `Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase`.
Owner phase ∈ {5 naming/API, 6 core engine, 8 build/test infra, 9 docs}.

Reality note confirmed this pass: ELIPS CPU core is mature; the gaps are concentrated in
API vocabulary, numpy/contract ergonomics, build/packaging infra, query-parallelism, test
breadth (recall/QPS/GPU-real/parity), and doc truth — not missing engine logic.

---

## 1. API naming / ergonomics

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| Three in-code vocab layers | FAISS=one canon (`add`/`search`/`Index`); Chroma=one canon (`add`/`query`/`Collection`) | 3 live layers: C++ `place`/`seek`/`Vault` (`elips.hpp:130,149,122`); Python modern `add`/`search`/`Collection` (`collection.py:939-958`); docs/site teach OLD `arena`/`ingest`/`probe`/`seek` (`_phase1_docs.md:40-49`) | blocker | L | Pick Chroma names as Python canon + FAISS names as C++ canon; demote `arena/ingest/probe/top/meta` to deprecated aliases that warn; purge old vocab from docs | 5 (+9 docs) |
| Clients | `Client()`, `PersistentClient(path=)`, `EphemeralClient`, `HttpClient` (`_faiss_3G:24-30`) | none — only `connect()`→`Engine` (`connect.py:58`); "`Client`/`PersistentClient` do not exist" (`_phase1_python.md:376`) | major | M | Add `Client`/`PersistentClient`/`EphemeralClient` thin wrappers over `connect()`; defer `HttpClient`/`Cloud`/async (YAGNI) | 5 |
| Client-side collection mgmt | `create_collection`/`get_collection`/`get_or_create_collection`/`delete_collection`/`list_collections` (`_faiss_3G:36-41`) | `Engine.collection()`/`collection_names()` only (`engine.py:166,170`); no get-or-create/delete/list under Chroma names | major | M | Add the five Chroma collection-mgmt methods on the client, mapped onto `Engine` | 5 |
| `top=` → `n_results` | query top-k arg is `n_results` (default 10) (`_faiss_3G:97,340`) | `top=10` on probe/search (`collection.py:201-203`) | major | S | Rename to `n_results`; accept `limit` alias; keep `top` as deprecated alias | 5 |
| `meta` → `metadatas` | `metadatas` = list of dicts (`_faiss_3G:96`) | high-level `meta=`, native `data=` (`collection.py:197`; `_phase1_python.md:373`) | major | S | Rename to `metadatas`; deprecate `meta`/`data` | 5 |
| Unified `query`/`get` signature | `query(query_embeddings=,query_texts=,n_results=,where=,where_document=,include=)`, `get(ids=,where=,where_document=,limit=,offset=,include=)` (`_faiss_3G:57-59`) | split `probe`/`search`/`probe_hybrid`/`sweep`/`pull`; `where=` yes, but no `where_document`, no `include=`, no `n_results` (`collection.py:201-206`) | major | M | Unify into Chroma-shaped `query`/`get`; add `include=[...]` projection + `query_embeddings`/`query_texts` | 5 |
| `where` operators | dict `$eq $ne $gt $gte $lt $lte $in $nin $and $or` (`_faiss_3G:107-116`) | `Filter` builder + EQL only (`Filter.hpp`; `search/__init__.py`); no `$`-dict surface | major | M | Add dict→`Filter` translator for Chroma `$`-operators (ELIPS `Comparator` already covers eq/ne/lt/le/gt/ge, `Filter.hpp:13`) | 5 |
| `where_document` | `$contains $not_contains $regex $not_regex`, case-sensitive (`_faiss_3G:132-134`) | lexical hybrid exists (`seek_hybrid`) but no document-content filter dict | major | M | Map `$contains`/`$not_contains`→`has_substring`; defer `$regex`/`$not_regex` | 5/6 |
| FAISS Index API in Python | `Index` class, `index_factory(d,desc)`, `read_index`/`write_index` (`_faiss_3A:147,192`) | none in Python — only `Collection` + native `Vault`; no `index_factory`/`read_index`/`write_index` (`00_inventory.md:32`) | major | M | Expose FAISS-recognizable `Index` facade + curated `index_factory` tokens (`Flat`,`IVF…,PQ…`,`HNSW…`,`IDMap`) + `read_index`/`write_index` API shape | 5 (API) / 6 (factory impl) |
| `upsert` | universal idempotent write (`_faiss_3G:298`) | `add` already replaces-by-id (`Database.cpp:895-926`); no named `upsert` | minor | S | Add `upsert` alias (ELIPS `add` semantics already overwrite) | 5 |
| `count` / `peek` | `count()`, `peek()`=first 10 (`_faiss_3G:62-63`) | `count()` yes (`collection.py:196`); no `peek()` | minor | S | Add `peek(n=10)` | 5 |
| distance/space config | `configuration={"hnsw":{"space":"l2|ip|cosine"}}` (`_faiss_3G:148-162`) | `connect(metric="cosine", index="graph")` kwargs (`connect.py`) | minor | S | Keep kwargs; optionally accept Chroma `configuration=` dict | 5 |
| C++ Index base contract | `add`/`search`/`reset`/`train`/`reconstruct`; `SearchParameters{sel}`+`IDSelector` family (`_faiss_3A:30-54,233-272`) | `IndexPort::insert/remove/search` (`IndexPort.hpp:28-36`); no `SearchParameters`/`IDSelector`; EQL/`Filter` not exposed as a selector | major | L | Rename/alias C++ verbs toward FAISS; add `SearchParameters`+`IDSelector`, back EQL predicate by an `IDSelector` for one filtered-search path | 5/6 |

---

## 2. Validation / errors

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| Typed exception taxonomy | FAISS throws typed; Chroma has `NotFoundError`/validation errors | registered: `ElipsError`/`DimensionMismatch`/`InvalidVector`/`ConfigError`/`NotFound`/`StorageError`/`LockConflict`/`ParseError` (`elips_python.cpp:372-386`). **Missing** `NotTrainedError`, `CollectionNotFoundError`, `BackendUnavailableError` (confirmed absent) | major | S | Add the three missing typed exceptions; wire `GpuError::backend_unavailable` (`elips_python.cpp:1291`) → `BackendUnavailableError`; `NotFound`→`CollectionNotFoundError` on unknown vault | 5 (names) / 6 (wire) |
| numpy dtype float32 enforcement | `py::array_t`/`ascontiguousarray(float32)`, zero-copy pointer bridge (`_faiss_3E:105-120`) | **no numpy interface** — vectors enter as `py::iterable`, element-wise `cast<float>()` (`elips_python.cpp:89-95,2351-2374,2480-2487`); float64 arrays silently narrowed, no zero-copy, per-element Python iteration | major | M | Switch bindings to `py::array_t<float, c_style\|forcecast>`, pre-alloc `D`/`I` outputs, document the silent cast | 6 (bindings) / 8 |
| numpy C-contiguity contract | enforced/forced at boundary (`_faiss_3E:108-118`) | not applicable — no array interface, so no contiguity concept or fast path (`elips_python.cpp:89-95`) | major | M | Same as above (`c_style\|forcecast` folds contiguity in) | 6/8 |
| NaN/Inf policy | FAISS: no default reject (has NaNEuclidean metric) | ELIPS **rejects** on insert and query: `all_finite()` (`Database.cpp:89-93`) → `InvalidVector` in `Vault::prepare` (`Database.cpp:827-828`) and query path (`Database.cpp:1889-1890`); also KMeans training (`KMeans.cpp:85-87`) | minor | S | Keep reject policy; declare it at the Python boundary and document it (not currently documented) | 6/9 |
| `k > ntotal` semantics | legal; unused slots padded with label `-1` + sentinel distance (`_faiss_3A:62-67`) | **clamps** to `min(k,n)`, returns fewer, no `-1` pad: `ExactIndex.cpp:80-85`, `HierarchicalGraphIndex.cpp:368,374-376` | minor | S | Document ELIPS behavior; add `-1` padding only if/when the FAISS `Index` API is exposed (contract match) | 6 |
| Duplicate-id policy | Chroma `add`: re-adding existing id is **ignored** (no-op); `upsert` overwrites (`_faiss_3G:84`) | ELIPS `add`/`place` **overwrites** (remove-then-insert, `Database.cpp:895-926`; `ExactIndex::insert` itself append-only, `ExactIndex.cpp:24-34`) | major | S | Decide: make `add` Chroma-faithful (ignore/err on dup) + `upsert` overwrite, OR document `add`=`upsert`. Pick one and document | 5 |
| Not-trained enforcement | `is_trained` flag; search-before-train guarded (`_faiss_3A:33,280`) | HNSW/exact need no training; `quantize()` throws `ConfigError` if no codec (`Database.cpp:1470`); no `is_trained`/`NotTrainedError` | minor | S | Add `NotTrainedError` for the quantized-search-before-train path; expose an `is_trained`-style flag on the Index facade | 6 |
| numpy contiguity error vs silent copy | documented silent `ascontiguousarray` copy (`_faiss_3E:118`) | silent narrowing with no doc note (`elips_python.cpp:92`) | minor | S | Adopt FAISS stance (silent copy) + one-line docstring note | 6/9 |

---

## 3. Build system / packaging

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| Opt-level / SIMD matrix | `FAISS_OPT_LEVEL` generic/avx2/avx512/avx512_spr/sve/**dd** → per-opt-level libs or runtime dispatch (`_faiss_3F:31,60-86`) | none — NEON hand-kernels only (`Metrics.cpp:43-69`), no `OPT_LEVEL`, no x86 SIMD tiers, no dd (CMake agent: absent) | major (x86) / minor (Apple-first) | M/L | Add `ELIPS_OPT_LEVEL` (generic/neon/avx2), NEON mandatory baseline; defer `dd` runtime dispatch (`# ponytail: per-tier libs now, dd when wheel count hurts`) | 8 |
| BLAS/Accelerate selection | `FAISS_ENABLE_MKL`+`BLA_VENDOR`+OpenBLAS/Accelerate fallback; batched `sgemm` brute force (`_faiss_3F:93-103`, `_faiss_3E:34-62`) | **no** `find_package(BLAS/LAPACK/OpenMP)`, no Accelerate; brute force is scalar/NEON, no gemm (CMake agent: absent) | major (CPU perf) | M | Add one `gemm` abstraction → Accelerate (macOS) / OpenBLAS (else) for brute force; keep blocking constants as tunable knobs | 8 (wire) / 6 (use) |
| install / export package | `install(EXPORT faiss-targets)` + `faiss-config.cmake` + `CMakePackageConfigHelpers` + per-header install (`_faiss_3F:119-122`) | **none** — no `install()` of any kind, no `elips-config.cmake` (CMake agent: absent) | major (C++ consumability) | M | Add `elips-targets` export + `elips-config.cmake` + versioned header install (mirror FAISS convention) | 8 |
| GPU umbrella flags | `FAISS_ENABLE_GPU`/`_METAL`/`_ROCM`/`_CUVS`; FATAL on Metal-on-non-Apple (`_faiss_3F:33-42`) | `ELIPS_GPU_METAL=${APPLE}`, `ELIPS_GPU_CUDA/HIP=OFF`, derived `ELIPS_GPU_ENABLED` (CMake `:46-59`); no single umbrella | minor | S | Keep Metal-first; tidy naming; add FATAL-on-non-Apple-Metal guard | 8 |
| Latent GPU build breaks | — | CUDA/HIP CMake name nonexistent `.cu`/`.hip` (actual `.cpp`); `ELIPS_GPU_TESTS` adds missing `src/gpu_engine/tests/`; dup-library link warning (`00_inventory.md:80-83`; CMake agent) | major (blocks if CUDA/HIP enabled) | S | Fix source filenames or gate correctly; drop/ship the missing test dir; de-dup `elips_core`/`elips_gpu` link | 8 |
| Build backend | scikit-build-core, version scraped from CMake (`_faiss_3F:161-163`) | setuptools + manual CMake `build_ext` (`setup.py:91-159`); version hardcoded `1.1.0` (`pyproject.toml:7`) | major | M | Migrate to scikit-build-core (pybind11-friendly); single-source version from CMake | 8 |
| Wheel matrix / abi3 | cibuildwheel cp310-314, **abi3** single wheel (`Py_LIMITED_API`), cpu/gpu/gpu-cuvs split (`_faiss_3F:148-170`) | cibuildwheel 2.22.0 cp311-313, **single** `elips` wheel, **no** abi3, **no** cpu/gpu split (release.yml); pyproject advertises 3.14 but CI builds cp311-313 only (classifier/CI mismatch) | major | M | Adopt abi3 single wheel; keep one Metal+CPU macOS package initially (GPU-split only once a GPU package exists); fix 3.14 mismatch | 8 |
| Version single-source | regex-scraped from one place (`_faiss_3F:162`) | drift: Python `__version__` 1.0.0 vs CMake/pyproject 1.1.0 (`_phase1_python.md:7`; `00_inventory.md:18`) | minor | S | One version source (scikit-build-core scrape) | 8 |
| C++ standard | C++20 (`_faiss_3F:9`) | C++23 (CMake `:8`) — deliberate divergence, not a gap (`_faiss_3F:213`) | n/a | — | Keep C++23; ObjC++ at 17 like FAISS Metal | — |

---

## 4. Concurrency

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| Query-parallel search | search parallelized over queries via OpenMP; `nt=min(nq,hw)` (`_faiss_3E:94-101`) | **none** — no OpenMP, no thread pool, no `set_num_threads` anywhere in `src`/`include`/bindings; single-threaded search (verified: zero `omp`/`ThreadPool`/`hardware_concurrency` hits) | major (throughput) | M | Add query-parallel batch search over a `std::thread` pool / `std::execution::par`; mirror FAISS "never set global thread count internally" (`_faiss_3E:138`) | 6 (impl) / 8 |
| `set_num_threads` API | ambient `omp_get_max_threads`; `faiss.omp_set_num_threads(n)` (`_faiss_3E:99`) | absent (verified) | minor | S | Expose one `set_num_threads()` on the Index/Engine mirroring FAISS convention | 6 |
| Documented reader/writer contract | thread-safety documented (concurrent reads safe) | **process-level** single-writer/multi-reader IS documented (`lock-manager.md`, `storage.md:100-106`, `ADR-0005`, `ADR-0008`); **in-process** thread-safety of concurrent `Vault`/`Engine` calls under-documented (per-`Vault` `shared_mutex`, `_phase1_cpp.md:255`) | minor | S | Document the in-process thread-safety contract + GIL-release behavior | 9 |
| Shared-lock (multi-reader) tested | — | `LockMode::shared` exists + read-only opens take shared lock (`storage.md:100`), but `shared` **untested** in C++ (`_phase1_cpp.md:203`); doc class def omits the `LockMode` param (drift) | minor | S | Add a multi-reader shared-lock test; fix the doc signature | 8 (test) / 9 (doc) |
| libomp shipping pain | FAISS fights libomp in macOS wheels (`_faiss_3E:73,101`) | avoided — ELIPS has no OpenMP dependency (clean start) | n/a | — | Keep OpenMP-free; use std pool (`_faiss_3E:138`) | 6 |

---

## 5. Tests / benchmarks

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| SIFT1M / recall-vs-QPS methodology | `bench_gpu_sift1m`, `bench_hnsw`, `demo_auto_tune` Pareto; `.fvecs/.ivecs`; `knn_ground_truth`; warmup+seed (`_faiss_3H:220-260`) | **none** — only synthetic recall@10 vs in-process `ExactIndex` oracle (`BenchMain.cpp`, `hnsw_recall_test.cpp`, `quantized_index_test.cpp`); no SIFT1M/fvecs/groundtruth files, no QPS-vs-recall sweep (tests agent) | major | L | Port `evaluate()` (ms/query, recall@k) + SIFT1M loader + nprobe/ef sweep + warmup + seeded RNG + exact-Flat R@1==1.0 sanity gate | 8 |
| CPU-vs-GPU parity harness | `compareIndices`/`compareLists` 3-knob tolerance (`maxRelErr`/`pctMaxDiff1`/`pctMaxDiffN`) (`_faiss_3H:62-110`) | `tests/parity/CpuGpuRecallParityTest.cpp` exists but single `recall>=0.95` assert and `GTEST_SKIP` without a device (`_phase1_gpu.md:69`); no tolerance model | major | M | Formalize `compareIndices` 3-knob tolerance (Metal fp32: `maxRelErr≈6e-3`, `pctMaxDiff1≈0.10`, `pctMaxDiffN≈0.015`), run on M4 | 8 (+7 real kernel) |
| GPU real-kernel coverage | real GPU kernels executed in tests (`_faiss_3H:61-67`) | GPU unit tests inject `RecordingBackend` (CPU emulator, `tests/unit/gpu/TestRecordingBackend.hpp`); real Metal top-k is CPU `partial_sort`; dead `.metal` file not in CMake; nothing runs real Metal in CI (`_phase1_gpu.md:62-69`) | major | M/L | Wire the dead `.metal`, move top-k onto GPU, add a real-kernel test that runs on the M4 (not skipped) | 7 (impl) / 8 (CI) |
| C++ `place_many` / `seek_hybrid` tests | — | **absent in C++** — only Python (`test_bindings.py:362,305`); real bodies otherwise unverified in-tree (`_phase1_cpp.md:259,262`; tests agent) | minor | S | Add C++ tests for `place_many` and `seek_hybrid` lexical fusion | 8 |
| Quantized read-path depth | — | component tests present (`quantized_index_test.cpp`, `m5_quantization_persistence_test.cpp`) but `export_snapshot`/`import_snapshot`, index `compressed()` branch, `IndexFactory`, `WAL::append_insert_quantized`/`reset` untested (`_phase1_cpp.md:320`) | minor | S | Add snapshot/transfer + compressed-branch + factory tests | 8 |
| Docs-snippet execution | wheel smoke + `unittest discover` (`_faiss_3H:141-144`) | no snippet/doctest runner: `test_module_docstrings.py` only asserts `__doc__` non-empty (`:24-30`); CI `--doctest-modules` runs but package docstrings carry wrong examples (`_phase1_python.md:17`); no README/markdown execution | major (keeps docs honest) | M | Add doctest execution of corrected docstrings + a markdown-snippet runner | 9 |
| pytest config | configured (`_faiss_3F:143`) | **none** — no `[tool.pytest.ini_options]`, no `addopts`/markers/`conftest` (tests agent; `pyproject.toml`) | minor | S | Add pytest config (gpu/slow markers, addopts, seed pinning) | 8 |

---

## 6. Docs accuracy (scale of drift from `_phase1_docs.md`)

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| Website teaches OLD surface | docs match shipped API | site vocab counts: `vault` 666 / `arena` 396 / `seek` 212 / `erase` 91 / `probe` 78 / `ingest` 65 vs `Collection` 12 / `.add(` 1 / `.search(` 1; live `/docs/python-sdk` says "There is no add, ingest, search, get, or delete" — contradicts `collection.py:939-958` (`_phase1_docs.md:46-49,111`) | blocker | L | Rewrite site + prose to `Collection`/`add`/`search`/`get`/`delete` after Phase-5 naming lands | 9 |
| GPU over-claims | — | 10 Accepted GPU ADRs + `gpu-engine.md` present CUDA/HIP/SYCL/Vulkan as implemented (cuVS/CAGRA/rocBLAS/SPIR-V/speedups); reality = Metal partial only (`_phase1_docs.md:50-53,75-97`); SYCL/Vulkan already deleted in Phase 2 | major | M | Retire/rewrite GPU ADRs to "Metal real, CUDA/HIP experimental"; correct `gpu-engine.md` | 9 |
| EQL dialect split on site | one grammar | two incompatible EQL dialects: lowercase (correct) + uppercase SQL-like `SEARCH FROM…HYBRID_WEIGHT` (nonexistent) (`_phase1_docs.md:132,196-197`) | major | M | Delete the uppercase SQL-like dialect pages | 9 |
| Signature drift | ref generated from code | `connect(gpu=True)`, `engine.gpu_info()`, `ElipsInstance`, `SearchResult`, `Filter.ge()`, 3 conflicting `open()` sigs + invented kwargs, wrong `index_type`/`durability`/`dimension` defaults (`_phase1_docs.md:54-61,114-123`) | major | M | Regenerate the API reference from code post-Phase-5 | 9 |
| Undocumented-but-shipped | — | `Collection`/`Engine`/`add`/`search`/`compress`/`health`/`merge`, CLI `quantize`, `QuantParams`/`Codec` undocumented (`_phase1_docs.md:62-64,112-113`) | minor | S | Document the shipped surface | 9 |
| Roadmap/changelog stale | — | shared locks + ef-adaptive listed "planned" but shipped v1.1.0; GPU absent from roadmap (`_phase1_docs.md:170-172`) | minor | S | Refresh roadmap/changelog | 9 |
| Docstring mismatches | — | `Engine` docstring claims transactions it lacks; `elips.query` cites nonexistent `Engine.query`; orphan `exceptions.py` (`_phase1_python.md:403-423`) | minor | S | Fix docstrings; add `Engine.query`/`Engine.transaction()` or correct text | 5 (code) / 9 (doc) |

---

## Biggest gaps (ranked)

1. **Vocabulary chaos (blocker, Phase 5).** Three live in-code vocabularies plus a website that
   teaches a fourth and explicitly denies the shipped `add`/`search`/`get`/`delete`. This is the
   single defect that most blocks adoption; everything in §1 and §6 hangs off resolving it.
2. **No numpy array contract (major, Phase 6/8).** Bindings take `py::iterable` and coerce
   element-by-element (`elips_python.cpp:89-95`) — no `py::array_t`, no float32/C-contiguity
   enforcement, no zero-copy. Both an ergonomics gap (vs Chroma/FAISS numpy) and a real
   per-call performance cliff on large batches.
3. **No query parallelism (major, Phase 6).** Zero OpenMP/thread-pool/`set_num_threads` — search
   is single-threaded. FAISS's throughput story is query-parallelism; ELIPS has none.
4. **No build export + setuptools backend + no abi3/cpu-gpu split (major, Phase 8).** No
   `install(EXPORT)`/`elips-config.cmake`, setuptools+manual CMake with a hardcoded version, a
   single non-abi3 wheel, and a 3.14 classifier/CI mismatch. C++ is not consumable and the wheel
   matrix is behind FAISS.
5. **No recall/QPS/SIFT1M methodology + no real-GPU-kernel test (major, Phase 7/8).** Recall is
   only synthetic-vs-oracle; no ANN-benchmark dataset, no QPS sweep, GPU tests run a CPU emulator
   and the real `.metal` kernel is dead code. "Metal validated" cannot be claimed.
6. **FAISS Index API absent in Python (major, Phase 5/6).** No `Index`/`index_factory`/
   `read_index`/`write_index` — the most FAISS-recognizable surface is missing.
7. **Missing typed exceptions + divergent contracts (major, Phase 5/6).** No
   `NotTrainedError`/`CollectionNotFoundError`/`BackendUnavailableError`; `add` overwrites where
   Chroma `add` ignores; `k>ntotal` clamps where FAISS pads with `-1`.
8. **No BLAS/Accelerate CPU brute-force path (major, Phase 8).** No gemm/BLAS; brute force is
   scalar/NEON only — leaves large-batch CPU search well below FAISS.
9. **Docs actively wrong (blocker for release, Phase 9).** GPU ADRs overclaim four backends; a
   fabricated uppercase EQL dialect; pervasive signature/default drift. Must follow Phase-5
   naming.

**Lazy-win cluster (Effort S, do early):** `n_results`/`metadatas` renames, `upsert`/`peek`
aliases, the three missing exceptions, pytest config, version single-source, `place_many`/
`seek_hybrid` C++ tests, shared-lock test — each is S and unblocks later phases cheaply.

