# 01 — FAISS Study (consolidated, with file:line references)

Studied tag/commit: **FAISS v1.15.1** @ `75c755c5a1fc854756aaf3bd1131213aeb21d8d1` (cloned to `/tmp/faiss`, 2026-10-10).

Each section ends with an explicit **adopt / adapt / skip** decision + `file:line` citations.
Corrections to this prompt's memory of FAISS go here (reality wins).
Full per-section detail with every `file:line` lives in `_faiss_3A.md` … `_faiss_3I_metal.md`;
this file consolidates the decisions.

## Reality corrections (the prompt's memory of FAISS was wrong here — reality wins, rule #7)

1. **FAISS v1.15.1 HAS a Metal backend** (`faiss/gpu_metal/`, ~7700 LOC) — overrides mission
   ground rule #8 ("Metal has no FAISS precedent"). Flat & IVFFlat do distance **and top-k on
   the GPU** (`topk_threadgroup_2048`, `MetalDistance.metal:244`); IVFPQ is a CPU stub. See ADR-R1
   in `04_architecture.md` and `_faiss_3I_metal.md`. This is the precedent for ELIPS Phase 7.
2. **GPU temp memory is a tiered cap, not a fraction**: ≤4 GiB→512 MiB, ≤8 GiB→1 GiB, else 1.5 GiB
   (`StandardGpuResources.cpp:180-203`). Metal backend uses **no arena at all** (per-call shared buffers).
3. **Max k on GPU = 2048** (CUDA; 1024 below sm_90; cuVS 16384) — a register/shared-mem budget,
   must be re-derived for Apple, not copied (`DeviceDefs.cuh:39`). Metal's is also 2048 (`MetalKernels.h:30`).
4. **FAISS requires C++20, not C++23** (`CMakeLists.txt:83`) — ELIPS's C++23 is a deliberate divergence.
5. **`FAISS_OPT_LEVEL` has 6 values** incl. `dd` (runtime Dynamic Dispatch), the pip default; CMake default `generic`.
6. **FAISS has no cosine metric** — it's IP on L2-normalized vectors (`fvec_renorm_L2`). Only IP+L2 are universal.
7. **`index_cpu_to_all_gpus` / `get_num_gpus` are Python/SWIG-level**, not core C++. `k>ntotal` legal → pads label -1.
8. Minor: `demo_ivfpq_indexing_gpu` is not a build target; `bfKnnBatch` is actually `bfKnn_tiling`.

## 3A — Core Index API (CPU) → `_faiss_3A.md`
`Index` has only 3 pure virtuals (`add`/`search`/`reset`, `Index.h:164,213,272`); rest default to
no-op/throw. Row-major `float*`, `idx_t=int64`, L2 returns **squared** distance. `k>ntotal` pads -1.
`SearchParameters{IDSelector*}` is the thread-safe per-call filter. Serialization = per-class fourcc, no magic/version.
**ADOPT:** Index contract, Flat/IDMap names, SearchParameters/IDSelector. **ADAPT:** factory strings, I/O
(add a real magic+version header, keep deserialization size caps). **SKIP:** NumericType/_ex, Panorama, binary, SVS.

## 3B — IVF & quantization → `_faiss_3B.md`
Defaults to mirror: `nprobe=1`, `max_codes=0`, `by_residual=true`; PQ `code_size=(nbits·M+7)/8`;
SQ `QT_8bit`+`RS_minmax`; OPQ `niter=50`; clustering `niter=25,nredo=1,seed=1234,max_points=256`.
Surprises: IP metric forces `spherical=true` (`IndexIVF.cpp:194`); PQ sub-quantizer always trained on
residuals when `by_residual`; FAISS bit-packs 4/6-bit PQ (ELIPS is 8-bit only).
**ADOPT** ELIPS's clearer names (`n_lists`/`n_probe`/`pq_dim`/`pq_bits`), record FAISS aliases. **ADAPT:**
add precomputed per-list centroid table for IVFPQ+L2; adopt `256·ksub` training cap. **KEEP** ELIPS's
deterministic k-means init over k-means++. **SKIP:** polysemous, parallel_mode, SDC, TurboQuant/EDEN.

## 3C — HNSW / graph → `_faiss_3C.md`
Params on the `HNSW` struct: `efConstruction=40`, `efSearch=16`; `M` baked into
`cum_nneighbor_per_level` (2M on L0, M above) and immutable after first add (`HNSW.cpp:39-45`).
Diversity-heuristic neighbor pruning; per-query `efSearch` override via `SearchParametersHNSW`.
**ADOPT** runtime `efSearch`; **map** ELIPS `GraphParams` (`max_connections`//M, `ef_construction`,
`ef_search`) — keep ELIPS's descriptive names, record M/efc/efs as aliases. **SKIP** IndexNSG, 2-level.

## 3D — GPU architecture (CUDA path) → `_faiss_3D.md`
`GpuIndexConfig` has 3 fields (device/memorySpace/use_cuvs); `IndicesOptions` 32/64-bit storage is
backend-agnostic. `GpuIndex : faiss::Index` → interchangeable; 3-method subclass contract
(`addImpl_`/`searchImpl_`/`addImplRequiresIDs_`); eager `FAISS_THROW` on unsupported ops. Resources =
shared object, per-device state, LIFO bump stack, **no mutexes** (ordering via streams).
**ADOPT:** Index-interchangeability + config shape + typed-throw-on-unsupported + bfKnn primitive +
parity test method (rel-err + rank-agreement + recall≥0.9). **ADAPT:** resources→`MTLHeap`+queue,
cloner single-GPU first. **REDESIGN (algorithms adopt):** k-select→simdgroup, GEMM via tiled kernel.
All CUDA constants (max-k 2048, 256 MiB pages, block sizes) are **calibration knobs**, re-tune on Apple.

## 3E — Distances / metrics / utilities → `_faiss_3E.md`
Only IP(0)/L2(1) universal; `is_similarity_metric`=IP+Jaccard selects min/max heap. knn uses
sequential-SIMD below `nx*d<128000` else BLAS `sgemm`; L2 = `‖x‖²+‖y‖²−2⟨x,y⟩` with negative-clamp.
Runtime SIMD dynamic-dispatch (`SIMDLevel`, `FAISS_SIMD_LEVEL`). numpy: zero-copy requires **float32 +
C-contiguity** (wrappers call `ascontiguousarray`, pre-alloc D float32 / I int64).
**ADOPT:** IP/L2+heap direction, cosine=IP-on-normalized (+zero-norm skip), gemm L2 reconstruction,
numpy float32/C-contig/int64 contract. **ADAPT:** Fortran `sgemm_`→row-major `cblas_sgemm` (Accelerate/
OpenBLAS + Metal), OpenMP→C++ pool. **SKIP:** scipy metric family, full AVX/SVE dispatch ladder.

## 3F — Build system & packaging → `_faiss_3F.md`
Options to mirror: `FAISS_ENABLE_GPU/ROCM/CUVS/PYTHON/C_API`, `BUILD_TESTING`, `BUILD_SHARED_LIBS`,
`FAISS_USE_LTO`, `FAISS_OPT_LEVEL` (generic/avx2/avx512/avx512_spr/sve/**dd**), `FAISS_ENABLE_MKL` (ON),
`CMAKE_CUDA_ARCHITECTURES`, `CUDAToolkit_ROOT`. Per-opt-level targets (`faiss`/`faiss_avx2`/…) from one
src list, non-matching `EXCLUDE_FROM_ALL`. Flow: configure→`make faiss`→swig+setup.py→install→CTest/pytest.
**ADOPT:** Metal build chain + install/export conventions, the layered flow, demo+test command structure,
cpu/gpu package split rationale. **ADAPT:** SWIG→pybind11, MKL→Accelerate on macOS, options→`ELIPS_ENABLE_*`
+ `ELIPS_OPT_LEVEL` (generic/neon/avx2/avx512). **SKIP (now):** SVS/cuVS, GPU_STATIC, conda gpu-split, `dd` (defer).

## 3G — Chroma & DB conventions → `_faiss_3G.md`
Chroma is the mirror target: `Client()` (ephemeral) / `PersistentClient(path=)` / `HttpClient`;
`create/get/get_or_create/delete/list_collections`; collection `add/upsert/get/update/delete/query/count/
peek/modify`; args `ids/embeddings/documents/metadatas/n_results/where/where_document/include`; `where`
ops `$eq $ne $gt $gte $lt $lte $in $nin $and $or`; `where_document` `$contains/$not_contains/$regex`.
RECENT: distance config moved to `configuration={"hnsw":{"space":l2|ip|cosine}}` (l2 default). Cross-DB:
only `upsert` is truly universal; `limit` beats `n_results` 3-of-4 vendors.
**ADOPT** full Chroma Collection surface + `Client`/`PersistentClient` (ELIPS lacks these today). **ADAPT:**
keep `n_results` but **alias `limit`**. **SKIP:** vendor-specific extras.

## 3H — Testing & benchmarking → `_faiss_3H.md`
One aggregate GoogleTest binary via `gtest_discover_tests`; GPU correctness = live CPU `IndexFlat` oracle
compared in-process (`compareIndices`/`compareLists`). Three tolerance knobs: `maxRelativeError` (6e-5),
`pctMaxDiff1` (0.1), `pctMaxDiffN` (0.005), overridden per (index,dtype,dim). `evaluate()` = (ms/query,
n-recall@1) with warmup + exact-Flat R@1==1.0 sanity gate. Ground truth = shipped `.ivecs` or exact blockwise.
**ADOPT:** oracle+three-knob parity helper, `evaluate()` methodology, SIFT1M+GloVe (skip billion-scale),
one suite parametrized over a CPU/Metal/CUDA backend fixture (CUDA build-only via `CMAKE_CROSSCOMPILING`).
Start Metal fp32 at 6e-3 / 0.10 / 0.015.

## 3I — FAISS Metal backend (THE Phase-7 precedent) → `_faiss_3I_metal.md`
State: `MetalIndexFlat` & `MetalIndexIVFFlat` do distance **and top-k on GPU**; `MetalIndexIVFPQ` is a CPU
stub. Resource layer is thin (1 device + 1 command queue, all `StorageModeShared`, no arena). Kernels:
hand-written tiled GEMM `l2_squared_matrix`/`ip_matrix` (not MPS), `topk_threadgroup_<K>` (256-thread
register-prefilter → threadgroup bitonic, **smaller-index tiebreak → exact CPU parity**, max K 2048),
`compute_norms`+`l2_with_norms`, IVF `ivf_scan_list*`+`ivf_merge_lists` with `nprobe*k≤1024` exactness guard.
metallib chain: `xcrun metal→.air→metallib` + 3-tier runtime discovery (env→compiled-in→dladdr). Python =
opaque `void*` holder with CUDA-identical symbol names. Tests: Flat/IVFFlat **exact label parity** (1e-5/1e-4);
IVFPQ only >50% recall.
**ADOPT:** `topk_threadgroup` (ELIPS's #1 missing artifact — replaces its CPU `std::partial_sort`), tiled
GEMM, `MetalIndexFlat::search` skeleton, metallib build chain, exact-parity test bar, shared-memory zero-copy
cloner. **ADAPT:** add a tiny reusable temp buffer (2× growth, 256 MiB tile budget), IVFFlat machinery
(drop env-var zoo), Python opaque holder. **SKIP:** IVFPQ GPU (no precedent), SQ4/SQ6 dead helpers, MPS link,
`MetalAllocType` pool.

---

## Consolidated decisions feeding later phases

- **Phase 5 (naming/API):** C++ Index API mirrors `Index`/`IndexFlat`/`IndexIVFFlat`/`IndexIVFPQ`/`IndexHNSW`
  + `index_factory` + SearchParameters/IDSelector (3A). Python Collection API mirrors Chroma incl. new
  `Client`/`PersistentClient` + `n_results`(+`limit` alias) (3G). Keep ELIPS's descriptive index-param names,
  record FAISS aliases (3B/3C). cosine = IP-on-normalized (3E).
- **Phase 6 (core):** add magic+version to the index serialization header (3A); precomputed centroid table
  for IVFPQ+L2 + `256·ksub` training cap (3B); numpy float32/C-contiguity/int64 boundary contract (3E).
- **Phase 7 (GPU, Metal-first):** adapt FAISS's Metal backend directly (3I) — port `topk_threadgroup` to replace
  ELIPS's CPU top-k, wire the metallib chain to revive the dead `.metal`, keep shared-storage zero-copy cloner;
  mirror the CUDA layering principles (3D) for the GpuResources/GpuIndex/Cloner shape; max-k and tile sizes are
  Apple calibration knobs. GPU indexes are `Index` subclasses that throw on unsupported ops.
- **Phase 8 (build):** `ELIPS_ENABLE_*` + `ELIPS_OPT_LEVEL`, Accelerate on macOS, pybind11, the metallib
  xcrun chain, install/export package config (3F).
- **Phase 8/test matrix:** CPU-Flat oracle + three-knob parity, SIFT1M+GloVe, one backend-parametrized suite,
  Metal fp32 tolerances 6e-3/0.10/0.015, exact label parity for Flat top-k (3H/3I).
