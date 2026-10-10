# 3E — FAISS Distances, Metrics & numpy/threading utilities

Source: **FAISS v1.15.1** @ `/tmp/faiss` (read-only, not built). ELIPS target: C++23, Metal-first, Accelerate/OpenBLAS on CPU.
Every claim cites `file:line`. Reality-vs-prompt corrections flagged with **⚠ CORRECTION**. Each item ends with **adopt / adapt / skip**.

---

## 1. Metric types

`enum MetricType` (`faiss/MetricType.h:29-49`):

| value | int | notes |
|---|---|---|
| `METRIC_INNER_PRODUCT` | 0 | max inner product (similarity) |
| `METRIC_L2` | 1 | **squared** L2 (not sqrt'd) |
| `METRIC_L1` | 2 | cityblock |
| `METRIC_Linf` | 3 | infinity |
| `METRIC_Lp` | 4 | p from `metric_arg` |
| `METRIC_Canberra` | 20 | scipy family (gap 5→19 reserved) |
| `METRIC_BrayCurtis` | 21 | |
| `METRIC_JensenShannon` | 22 | |
| `METRIC_Jaccard` | 23 | similarity: `sum min / sum max` |
| `METRIC_NaNEuclidean` | 24 | L2 ignoring NaNs |
| `METRIC_GOWER` | 25 | mixed numeric/categorical |

- **Broad support**: only `INNER_PRODUCT` and `L2` are supported by *all* indices. The L1/Linf/Lp + scipy family work **only on flat/brute-force indices** via `knn_extra_metrics` (`MetricType.h:21-25` doc comment; dispatched in `extra_distances.cpp`).
- `is_similarity_metric()` → true for `INNER_PRODUCT` and `Jaccard` only (`MetricType.h:56-59`); this selects min-heap (CMin) vs max-heap (CMax) throughout.
- Validation: `metric_type_from_int()` throws outside `[0,4] ∪ [20,25]` (`MetricType.h:63-70`); `metric_type_count()` (`:73-76`).

**ELIPS: ADOPT** the `INNER_PRODUCT`/`L2` core + the `is_similarity` → heap-direction distinction. **ADOPT cosine = IP on L2-normalized vectors** (FAISS has no cosine metric; it is IP after `fvec_renorm_L2`, see §3). **SKIP** the scipy family (Canberra/BrayCurtis/JensenShannon/Jaccard/NaNEuclidean/Gower) — YAGNI for a vector-search engine; add a single one later only on explicit request. **ADAPT** L1/Linf to a compile-time-dispatched flat-only path *if* a user needs them (`with_metric_type`, `extra_distances.h:72-108`).

---

## 2. knn_L2sqr / knn_inner_product — signatures, blocking, BLAS

Signatures (`faiss/utils/distances.h:390-456`): both a heap-array form (`float_minheap_array_t`/`float_maxheap_array_t`) and a raw-pointer form (`float* distances, int64_t* indexes`). L2 form takes an optional precomputed `const float* y_norm2`.

### Two-path strategy (per-vector SIMD vs BLAS)
Decision in `Run_search_inner_product::f` / `Run_search_L2sqr::f` (`distances.cpp:559-599`):
```
if (sel || ny == 0 || nx*d < distance_compute_blas_threshold)  → sequential (per-vector SIMD)
else                                                            → BLAS (sgemm)
```
Tunables (`distances.cpp:607-610`): `distance_compute_blas_threshold = 128000`, `blas_query_bs = 4096`, `blas_database_bs = 1024`, `min_k_reservoir = 100`.

### BLAS path
- IP: single `sgemm_("Transpose","Not transpose", ...)` per (query-block × db-block) computing the dot-product block (`distances.cpp:400-413`).
- L2: same `sgemm` for dot products, then reconstruct `dis = |x|² + |y|² − 2·⟨x,y⟩` (`distances.cpp:491`), with **negative clamp to 0** for roundoff on identical vectors (`:498-500`).
- Norm precompute: `fvec_norms_L2sqr(x_norms,...)` always; `y_norms` computed only if caller didn't pass `y_norm2` (`distances.cpp:443-453`).
- BLAS declared as raw Fortran `extern "C" int sgemm_(...)` with `FINTEGER` (= `long`) args, column-major transpose convention (`distances.cpp:30-52`).
- `Top1` (k=1) L2 has a fused specialization bypassing the norm/clamp loop (`distances.cpp:530-557`, `exhaustive_L2sqr_fused_cmax`).

### Sequential path
Per-query OpenMP loop, inner loop calls `fvec_inner_product<SL>` / `fvec_L2sqr<SL>` with `IDSelector` filtering (`distances.cpp:285-362`).

### ⚠ CORRECTION — new "database-parallel" path (not in older FAISS)
When `nx < nthreads` and `ny ≥ max(10000, nthreads·1024)` and no selector, FAISS parallelizes over **database segments** instead of queries, each thread keeping its own per-query heaps then merging (`distances.cpp:612-773`, gate `should_use_db_parallel` `:760-773`). Fixes the "few queries, huge DB, idle cores" case.

### IDSelector handling
`IDSelectorRange` → shifts `y`/`ny` and clears sel (`distances.cpp:786-792`); `IDSelectorArray` → delegates to `knn_*_by_idx` (`:793-797`). Any other selector forces the sequential path.

**ELIPS: ADOPT** the `|x−y|² = |x|²+|y|²−2⟨x,y⟩` + batched-gemm scheme, norm precompute, and negative clamp — this is the correct, fast CPU brute-force. **ADAPT the BLAS abstraction**: do *not* copy the Fortran `sgemm_`/`FINTEGER`/column-major-transpose idiom. Wrap one `gemm(...)` → `cblas_sgemm` (Accelerate `vecLib` on macOS, OpenBLAS elsewhere) with **row-major** API so you skip the transpose gymnastics. On Metal the same batched-IP = one matmul (`MPSMatrixMultiplication`/MLX) behind the same abstraction. **ADAPT** the blocking constants as *calibration knobs* (threshold/query_bs/db_bs) — keep them tunable, the 128000/4096/1024 values are hardware-era-specific. **SKIP initially** the db-parallel path (it is new and adds real complexity); add it only when "nq < cores, large DB" is a measured workload.

---

## 3. Normalization & fvec helpers

- `fvec_renorm_L2(d, nx, x)` (`distances.cpp:269-275`): in-place L2 normalize; **branches** to a no-OMP variant when `nx ≤ 10000` else OMP (`:238-267`). **Skips zero-norm vectors** (`if (nr > 0)`), avoiding div-by-zero. The split exists to dodge a real libomp crash with `nx==1` (`:227-237`, documented).
- `fvec_norms_L2` / `fvec_norms_L2sqr` (`distances.cpp:201-225`): OMP `if (nx > 10000)`.
- Public scalar wrappers `fvec_L2sqr`, `fvec_inner_product`, `fvec_norm_L2sqr`, `fvec_L1`, `fvec_Linf` all forward to `*_dispatch` (`distances.cpp:60-78`).
- `inner_product_to_L2sqr` (`distances.cpp:1178-1191`): `|x−y|² = nr1+nr2−2·ip`; header note says it exists because Torch's version is slow/non-threaded (`distances.h:240-242`).

**ELIPS: ADOPT** `fvec_renorm_L2` semantics exactly for the **cosine path** — normalize at add time and query time, skip zero vectors. The zero-skip is a correctness corner, do not drop it. **ADAPT** the `nx>10000` OMP threshold to ELIPS's threading primitive (see §5). The libomp-crash workaround is OpenMP-specific; irrelevant once ELIPS drops OpenMP.

---

## 4. SIMD dispatch — ⚠ MAJOR CORRECTION vs prompt

The prompt's mental model (`FAISS_OPT_LEVEL` picks AVX2/AVX512/NEON/SVE at compile time) is the **old** scheme. v1.15.1 adds a **runtime Dynamic Dispatch (DD)** layer:

- `enum class SIMDLevel { NONE, AVX2, AVX512, AVX512_SPR, ARM_NEON, ARM_SVE, RISCV_RVV, AVX512_VPOPCNT }` (`simd_levels.h:19-37`).
- Every `fvec_*` is a template `<SIMDLevel>` (`distances.h:31-104` etc.); public fn calls `*_dispatch`; the hot loops instantiate via a `with_simd_level([&]<SIMDLevel SL>(){...})` lambda (`distances.cpp:206, 304, 345, 944...`).
- Per-arch kernels live in `faiss/utils/simd_impl/`: `distances_aarch64.cpp` (NEON, `vld1q_f32`/`vfmaq` etc., `:15-120`), `distances_autovec-inl.h` (scalar `NONE`), plus AVX2/AVX512 files. `distances_simd.cpp` is only the thin `SIMDLevel::NONE` TU (`distances_simd.cpp:17-172`).
- **Build knob is still `FAISS_OPT_LEVEL`** (`faiss/CMakeLists.txt:88`, default `"generic"`), values `avx2|avx512|avx512_spr|sve|dd`. `FAISS_OPT_LEVEL=dd` → defines `FAISS_ENABLE_DD` and compiles *all* levels into one lib (`CMakeLists.txt:541-548`).
- **Runtime override**: env `FAISS_SIMD_LEVEL` (DD builds only) overrides auto-detect; if the requested level isn't compiled in, it walks *down* to the nearest compiled level and warns (`simd_levels.cpp:186-213`). Auto-detect via CPUID/HWCAP (`:236+`).
- Static (non-DD) builds: `SINGLE_SIMD_LEVEL` resolves to the one compiled level at zero overhead; on ARM64, NEON is the mandatory baseline (`simd_levels.h:48-75`).

**ELIPS: SKIP** the DD machinery and the AVX/SVE/RISCV ladder — on Apple Silicon NEON is mandatory and the GPU does the heavy lifting. **ADAPT**: for CPU brute-force rely on Accelerate/OpenBLAS gemm (already hand-tuned SIMD) + compiler autovectorization at `-mcpu=apple-m1`/native for single-vector distances. A `FAISS_SIMD_LEVEL`-style env toggle is only worth adding if a measurement demands it. Keep ELIPS distance kernels as plain C++23 templated on `float` and let the backend (Metal / BLAS) carry the vectorization.

---

## 5. OpenMP threading policy

- **knn (query-parallel)**: `nt = min(nx, omp_get_max_threads())`, `#pragma omp parallel num_threads(nt)` + `#pragma omp for` over queries (`distances.cpp:299-301, 340-342`).
- **knn (db-parallel)**: see §2; `nt = omp_get_max_threads()` threads split `ny` (`distances.cpp:630, 652-655`).
- **norms/renorm**: `#pragma omp parallel for if (nx > 10000)` (`distances.cpp:207, 220, 255`).
- **extra metrics**: query-parallel with periodic `InterruptCallback::check()` chunking (`extra_distances.cpp:107-135`).
- **IndexIVF**: `int parallel_mode = 0` (`IndexIVF.h:211`); mode 0 (the "95% codepath", `IndexIVF.cpp:319`) parallelizes over queries `nt = min(omp_get_max_threads(), n)` (`:379-384`); other modes parallelize over probed lists; `PARALLEL_MODE_NO_HEAP_INIT = 1024` bit flag (`IndexIVF.h:212`). `add` uses `#pragma omp parallel reduction(+:nadd)` (`IndexIVF.cpp:269`).
- **Convention**: FAISS never calls `omp_set_num_threads()` *inside* the library — it reads the ambient `omp_get_max_threads()` / `OMP_NUM_THREADS`. The caller sets thread count globally (`faiss.omp_set_num_threads(n)` in Python).

**ELIPS: ADOPT** the policy (parallelize over queries, thread count = `min(nq, hw_threads)`, never set the global thread count from inside the engine). **ADAPT the primitive**: use C++23 `std::thread`/a small thread pool (or `std::for_each(std::execution::par,...)`) instead of OpenMP — libomp is a known pain to ship in macOS wheels and conflicts with Accelerate's own threading. Expose one `set_num_threads()` on the ELIPS index mirroring FAISS's convention.

---

## 6. numpy interop (Python layer) — the contract ELIPS Python must enforce

Zero-copy pointer bridge:
- `swig_ptr(a)` (`faiss/python/swigfaiss.swig:1263-1332`): returns a **raw pointer into the numpy buffer** (no copy). Hard requirements enforced here: must be a numpy array; **`PyArray_ISCONTIGUOUS` or it raises `ValueError("array is not C-contiguous")`** (`:1278-1281`); dtype → SWIG pointer type map, `NPY_FLOAT32 → p_float` (`:1283-1309`), `NPY_INT64 → p_long/long_long` (`:1323-1328`).
- `rev_swig_ptr(ptr, size)` wraps a C pointer as a numpy view (then `.copy()` for ownership, e.g. `__init__.py:460-463`).

Enforcement at the method boundary (`class_wrappers.py`):
- `replacement_add` (`:289-309`): `n,d = x.shape; assert d == self.d; x = np.ascontiguousarray(x, dtype="float32"); self.add_c(n, swig_ptr(x))`.
- `replacement_search` (`:421-480`): same `ascontiguousarray(x, float32)` + `assert d == self.d`, `assert k > 0`; **pre-allocates outputs** `D = np.empty((n,k), float32)`, `I = np.empty((n,k), int64)` and fills them in place via `swig_ptr` (`:458-469`).
- `replacement_add_with_ids` (`:311-335`): ids forced to `int64`.
- Codes (PQ/binary) forced to `uint8` with an explicit `TypeError` instead of silent cast (`:33-44`).
- std::vector ↔ numpy: `vector_to_array` / `copy_array_to_vector` do a `memcpy` and **assert matching dtype** via `vector_name_map` (`array_conversions.py:111-163`).

So the real rules: **float32, C-contiguous, correct shape `(n,d)`** for vectors; `int64` ids/labels; `float32` distances. `ascontiguousarray` is a **silent copy only when needed** (no-op if already f32/C-contiguous) — that is the zero-copy fast path.

**ELIPS: ADOPT the contract verbatim** — enforce float32 + C-contiguity + `(n,d)` shape at the Python boundary, pre-allocate `D(float32)`/`I(int64)` outputs, pass by pointer. **ADAPT the mechanism**: ELIPS Python is almost certainly pybind11/nanobind, not SWIG. Use `py::array_t<float, py::array::c_style | py::array::forcecast>` in the signature — it enforces dtype+contiguity *at the call boundary* and is cleaner than SWIG's manual `swig_ptr` guard. Decide one of: (a) `forcecast`/`ascontiguousarray` = silent copy like FAISS (document it), or (b) raise on non-contiguous. Recommend (a) + a one-line docstring note, matching FAISS behavior so users aren't surprised. Keep int64 for ids/labels.

---

## Summary of decisions

| Item | Decision |
|---|---|
| IP + L2 core, similarity→heap-dir | ADOPT |
| Cosine = IP on `fvec_renorm_L2` vectors (+zero-skip) | ADOPT |
| scipy metric family (Canberra…Gower) | SKIP |
| L1/Linf/Lp flat-only | ADAPT (only if asked) |
| `|x−y|²=|x|²+|y|²−2⟨x,y⟩` + batched gemm + neg-clamp | ADOPT |
| Fortran `sgemm_`/FINTEGER/col-major | ADAPT → row-major `cblas_sgemm` (Accelerate/OpenBLAS) + Metal matmul, one `gemm` abstraction |
| Blocking constants (128000/4096/1024) | ADAPT as tunable knobs |
| db-parallel knn path | SKIP initially |
| Dynamic-dispatch SIMD / AVX/SVE ladder | SKIP (NEON mandatory on Apple; BLAS/Metal carry SIMD) |
| Query-parallel threading, no internal `set_num_threads` | ADOPT |
| OpenMP as the primitive | ADAPT → std::thread/exec pool |
| numpy float32 + C-contig + int64 ids, pre-alloc outputs, pointer pass | ADOPT contract |
| SWIG `swig_ptr` guard | ADAPT → pybind11 `c_style|forcecast` |
