# Phase 3H — FAISS Testing & Benchmarking Methodology (study)

Source: FAISS v1.15.1 at `/tmp/faiss` (read-only, not built). All citations are
`path:line` into that tree. ELIPS context: GoogleTest+CTest (C++) and pytest;
Apple M4 (Metal+CPU) is the only real GPU hardware, CUDA/HIP is compile-check
only. Goal: a FAISS-grade recall/QPS methodology and a CPU-vs-GPU parity
approach that fits `05_test_matrix.md` and the Phase 8 bench suite.

Verdicts are **ADOPT** (take as-is), **ADAPT** (take the idea, change for
ELIPS/Metal), **SKIP** (not worth it here).

---

## 1. C++ test layout & GoogleTest usage

**What's there.** One aggregate test binary. `tests/CMakeLists.txt:7-66` lists
~50 `test_*.cpp` into `FAISS_TEST_SRC`, then `add_executable(faiss_test ...)`
(`:103`). GoogleTest is vendored by `FetchContent` pinned to a commit
(`tests/CMakeLists.txt:119-124`, tag `58d77fa...` = release-1.12.1), with
`BUILD_GMOCK OFF` and `INSTALL_GTEST OFF` (`:125-126`) — gtest only, no gmock.
Registration is automatic: `gtest_discover_tests(faiss_test)`
(`tests/CMakeLists.txt:171`) turns every `TEST()` into its own CTest case.
Windows defers discovery to CTest time (`DISCOVERY_MODE PRE_TEST`, `:168`) and
cross-compiles skip discovery entirely (`:163-166`) — the binary just has to
link.

**Conditional compilation of the test list** (the interesting part):
- POSIX-only tests (`test_merge`, `test_ondisk_ivf`, `test_mmap`, ...) are added
  only `if(NOT WIN32)` (`tests/CMakeLists.txt:68-75`).
- SIMD-dispatch tests are gated on `FAISS_OPT_LEVEL STREQUAL "dd"` (dynamic
  dispatch build) (`:81-86`), and x86 raw-intrinsic tests
  (`test_simd_levels_x86_avx2/avx512`) are added only on x86 and compiled with
  per-file `-mavx2.../-mavx512...` flags via `set_source_files_properties`
  (`:87-114`).
- Optional-feature tests gated on a flag: `test_svs.cpp` only
  `if(FAISS_ENABLE_SVS)` (`:77-79`).

**Style.** Plain `TEST(Suite, Name)` + `EXPECT_*`/`ASSERT_*`, no fixtures for
most (`tests/test_utils.cpp:13` = the whole "version" test;
`tests/test_hnsw.cpp:222` `TEST(HNSW, Test_popmin)`). Helpers are tiny and
header-only: `tests/test_util.h:14-36` is just an RAII `Tempfilename` (mkstemp +
unlink in dtor). No heavyweight base fixtures, no mocking framework.

**ELIPS verdict.**
- **ADOPT** the single aggregate `elips_test` binary + `gtest_discover_tests` —
  ELIPS already does GoogleTest+CTest, this is the identical pattern (its 256
  CTest cases per `05_test_matrix.md` already come from this).
- **ADOPT** FetchContent-pin GoogleTest to an exact commit (reproducible CI; no
  "latest broke us").
- **ADAPT** the opt-level/arch gating to ELIPS's backend axis: gate
  Metal-specific test sources on the Metal-enabled flag the way FAISS gates
  `FAISS_ENABLE_SVS`/`FAISS_OPT_LEVEL`. For CUDA/HIP (compile-only here), copy
  the `CMAKE_CROSSCOMPILING` guard (`tests/CMakeLists.txt:163-166`): compile and
  link the GPU test binary but do **not** `gtest_discover_tests` it, so CI
  proves it builds without trying to run kernels on absent hardware.
- **SKIP** gmock — FAISS doesn't use it and neither should ELIPS; the existing
  `TestRecordingBackend.hpp` mock is hand-rolled, which is the right call.

---

## 2. GPU CPU-vs-GPU parity (the core of 3H)

FAISS's whole GPU-correctness strategy = **build a CPU `IndexFlat`/CPU index as
the reference, build the GPU index from the same vectors, compare search
results within tolerances.** The reference is never a precomputed file — it's a
live CPU index in the same process.

### 2a. The comparison primitive — `compareLists` / `compareIndices`

`faiss/gpu/test/TestUtils.h:66-108` declares `compareIndices()` (runs the same
query set through `refIndex` and `testIndex` then diffs) and `compareLists()`
(diffs two `(distance,index)` result tables). Three tolerance knobs, with these
**defaults** (`TestUtils.h:72-74`):

| knob | default | meaning |
|---|---|---|
| `maxRelativeError` | `6e-5` | per-element max relative distance error, `|a-b|/(0.5(|a|+|b|))` (`TestUtils.h:30-31`, impl `TestUtils.cpp:92`) |
| `pctMaxDiff1` | `0.1` | ≤10% of result slots may have *any* rank disagreement (off-by-1, off-by->1, or not-found) |
| `pctMaxDiffN` | `0.005` | ≤0.5% of result slots may be off by *more than one rank* |

The diff logic (`TestUtils.cpp:234-356`): for each query it builds a `index→rank`
map from the reference, then walks the test result list and classifies each
entry as `diff1` (rank differs by 1), `diffN` (differs by >1), or `diffInf`
(index absent from reference) (`:306-328`). It asserts:
1. per-element relative **distance** error ≤ `maxRelativeError`
   (`EXPECT_LE(relErr, maxRelativeError)`, `:333-341`);
2. total rank disagreements `(diff1+diffN+diffInf) ≤ numResults*pctMaxDiff1`
   (`:347-351`);
3. hard rank errors `diffN ≤ numResults*pctMaxDiffN` (`:355`);
4. returned indices within a query are **unique** (duplicate = serious error,
   `:299-305`) and that a `-1` ("no result") appears in the *same slot* in both
   (`:293-296`).

Rationale baked into comments: approximate + float rounding means you cannot
demand exact index equality; you bound (a) distance error tightly and (b) how
badly ranks may shuffle, loosely. Off-by-one at the tail is expected and
tolerated; gross reordering is not.

### 2b. Per-index-type tolerances (reality: defaults get overridden)

Tolerances are tuned per index and per dtype:
- **Flat**: fp32 `maxRelErr=6e-3`, fp16 `0.07` (`TestGpuIndexFlat.cpp:20-21`).
  For fp16 the rank tolerances are blown wide open (`pctMaxDiff1=0.99`,
  `pctMaxDiffN=0.65`, `TestGpuIndexFlat.cpp:104-108`) with a FIXME that fp16
  accumulators make rank bounds "useless" — so for fp16 they effectively only
  trust the distance relative-error bound.
- **IVFPQ**: `getCompareEpsilon()` = `0.035`, but `0.15` when `dim<=8` because
  near-zero L2 distances make relative error unstable
  (`TestGpuIndexIVFPQ.cpp:106-114`); `pctMaxDiff1` = `0.10` fp32 / `0.30` fp16;
  `pctMaxDiffN` = `0.02` fp32 / `0.05` fp16 (`:116-121`).

Takeaway: **tolerance is a function of (index type, dtype, dimensionality)**,
not one global number.

### 2c. How k / dims / db-sizes are parametrized

Not a fixed grid — **randomized per run inside an `Options`/`TestFlatOptions`
struct, resampled each `tries` iteration**, with a logged seed for repro
(`newTestSeed()`, `TestUtils.h:23`). Ranges from `TestGpuIndexIVFPQ.cpp:57-90`
and `TestGpuIndexFlat.cpp:46-62`:

| param | range | note |
|---|---|---|
| db size (`numAdd`) | `randVal(2000,5000)` | small, fast |
| dim | `randVal(50,800)` (Flat) / codebook-derived (IVFPQ) | |
| `numCentroids` | `sqrt(numAdd)` | |
| `nprobe` | `min(randVal(40,1000), numCentroids)` | |
| `numQuery` | `randVal(1,512)` (Flat) / `randVal(4,8)` (IVFPQ) | |
| `k` | **small on purpose**: `min(randVal(5,20), numAdd/40)` IVFPQ; fp16 Flat capped at `min(randVal(1,50),numVecs)` | big k + fp16 → too many tail diffs (`TestGpuIndexFlat.cpp:54-62`) |
| device | `randVal(0, numDevices-1)` | exercises multi-GPU |

Each `TEST()` loops `for (tries...)` to resample a few configs
(`TestGpuIndexFlat.cpp:111`).

### 2d. Python-side parity (where exactness *is* expected)

For paths that should be bit-exact (preassigned IVF add/search), Python asserts
**exact label equality + tight distance allclose**:
`np.testing.assert_array_equal(gpu_i, cpu_i)` and
`np.testing.assert_allclose(gpu_d, cpu_d, rtol=1e-5, atol=1e-5)`
(`test_gpu_index.py:70-71,121-122`); alt-distance Flat uses
`assert_array_equal(Inew,Iref)` + `assert_allclose(Dnew,Dref,rtol=1e-6)`
(`test_gpu_basics.py:174-181`). GPU k-means parity: `np.allclose(obj1,obj2)` on
the objective (`test_gpu_basics.py:134`).

**ELIPS verdict.**
- **ADOPT** the core pattern: live CPU index as oracle, GPU index from same
  vectors, compare in-process. This is exactly ELIPS's existing
  `CpuGpuRecallParityTest` (#255) idea — formalize it on this primitive.
- **ADOPT** the three-knob tolerance model (`maxRelErr`, `pctMaxDiff1`,
  `pctMaxDiffN`) as a reusable `compareIndices()` helper in ELIPS's gpu test
  dir. It is more honest than a single rtol because it separates *distance*
  error from *rank* churn.
- **ADAPT starting tolerances for Metal fp32**: use Flat fp32 `maxRelErr≈6e-3`,
  `pctMaxDiff1≈0.1`, `pctMaxDiffN≈0.015` (FAISS Flat defaults) as the opening
  bid, then widen per observed M4 behavior. Metal fp32 GEMM/reductions differ
  from CPU in reduction order, so expect to need Flat's `6e-3`, not the `6e-5`
  generic default. If ELIPS runs Metal fp16/bf16, copy FAISS's fp16 stance:
  trust distance rel-err (`~0.07`) and effectively drop the rank bound.
- **ADOPT** the "small k on purpose" rule and **ADOPT** logging a seed for
  randomized configs. But **ADAPT** parametrization to a *small fixed matrix*
  plus one randomized smoke, rather than FAISS's fully-random: fixed cases make
  `05_test_matrix.md` rows reproducible and reviewable; FAISS can afford pure
  randomness because of volume.
- **ADAPT** the dtype/dim-dependent epsilon (`dim<=8 → 0.15`): keep a tiny
  tolerance table keyed by (index, dtype) instead of one constant.
- **ADOPT** Python exact-equality assertions for any ELIPS path that is supposed
  to be deterministic CPU/Metal (e.g. a Flat/exact Metal kernel) — if ELIPS
  claims an exact kernel, assert `array_equal` on labels; if it can't pass that,
  it's not exact and `05_test_matrix.md`'s "real Metal" caution stands.

---

## 3. Python tests, random data, and the brute-force oracle

**Random data generators** (`tests/common_faiss_tests.py`):
- `get_dataset(d,nb,nt,nq)` — uniform `rand`, fixed `RandomState(123)` (`:82-87`).
- `get_dataset_2(...)` — "not completely random but still challenging": a 10-dim
  intrinsic ellipsoid projected into `d` dims then passed through `sin`, seed
  `1338` (`:90-105`). This is the generator used for accuracy tests because
  pure-uniform data is too easy/degenerate for ANN.
- `random_unitary` — Gaussian then L2-normalized (`:28-31`).
All seeded → deterministic.

**The oracle = exact `IndexFlat`.** Pattern (`test_index_accuracy.py:241-245`):
```python
gt_index = faiss.IndexFlat(d, mt); gt_index.add(xb)
gt_D, gt_I = gt_index.search(xq, 10)      # brute-force ground truth
...                                        # build approximate index
ninter = faiss.eval_intersection(I, gt_I) # overlap vs oracle
assert abs(ninter - self.ref_results[(mt,qname)]) <= 10   # tolerance band
```
Note the assertion style: not "recall > X" but "within ±10 of a recorded
golden number" — a regression guard, not an absolute-quality gate.

**Recall harness** (`common_faiss_tests.py:33-62`): `Randu10k` computes
`gt = dotprods.argmax(1)` (exact 1-NN by brute force), `evalres` returns
recall@{1,10,100} = fraction of queries whose true 1-NN lands in the top-rank.
Sanity: exact indexes must hit `e[1]==1.0` (`test_index_accuracy.py:44,50`);
approximate indexes get a floor assert (`assert e[1] > 0.2`,
`test_index_accuracy.py:60`). `omp_set_num_threads(4)` globally to keep runtime
sane (`common_faiss_tests.py:21`, comment: 100s→4s).

**ELIPS verdict.**
- **ADOPT** `IndexFlat`-as-oracle: ELIPS's exact/Flat index is the ground truth
  for every approximate index test, computed in-process. No golden files needed
  for small sizes.
- **ADOPT** `get_dataset_2`'s "structured but hard" generator — port it. Uniform
  random (ELIPS's current `rs.rand`) under-tests ANN quality.
- **ADOPT** seeded `RandomState` everywhere for determinism (ELIPS pytest should
  pin seeds).
- **ADAPT** the assertion style: use *both* an absolute floor (`recall@10 >
  threshold`, catches "it broke") and a golden-band regression (`abs(recall -
  recorded) < eps`, catches silent drift) — FAISS leans on the golden band;
  ELIPS's `05_test_matrix.md` wants absolute recall numbers too.
- **ADOPT** thread-pinning in tests for stable timing.

---

## 4. Benchmarks — recall@k + QPS methodology (Phase 8)

**The canonical loop** (`benchs/datasets.py:33-45`, `evaluate()`): time one
`index.search(xq, k)`, then
`recalls[i] = (I[:, :i] == gt[:, :1]).sum() / nq` for `i = 1,10,100`. Returns
`(ms_per_query, recalls)`. This is **n-recall@1**: "is the single true nearest
neighbor present in the top-n returned?" — the metric FAISS reports as `R@1`,
`R@10`, `R@100`. (The multi-NN variant uses `knn_intersection_measure`,
`contrib/evaluation.py:17-22`: mean `|intersect(I1[q], I2[q])| / rank`.)

**SIFT1M script** (`benchs/bench_gpu_sift1m.py`): load `.fvecs/.ivecs`, build
exact `GpuIndexFlatL2` first as a **sanity oracle** (R@1 must be `1.000` at every
k, `:48-52`), then an `IVF4096,PQ64` approximate index swept over
`nprobe = 1<<lnprobe` for `lnprobe in range(10)` (`:76-86`) printing ms/query +
R@1/R@10/R@100 per nprobe. Crucially a **warmup** `index.search(xq, 123)` runs
before timing (`:36,72`).

**HNSW script** (`benchs/bench_hnsw.py`): `DatasetSIFT1M` loader
(`:25-30`), sweep `efSearch ∈ {16,32,64,128,256}` (`:76-80`), report
`ms/query`, `R@1`, and `missing_rate` (`:44-50`).

**Recall-vs-QPS sweep** (`demos/demo_auto_tune.py`): the Pareto-frontier tool.
Builds a `OneRecallAtRCriterion(nq, 1)` with ground truth (`:66-69`), then
`ParameterSpace()` (CPU) or `GpuParameterSpace()` (GPU) `.explore(index, xq,
crit)` returns `OperatingPoints` = the recall-vs-time Pareto front (`:140-144`);
plots time ms/query (log y) vs `1-recall@1` (`:155-172`). **Same code toggled by
one `use_gpu` flag** that routes the factory output through
`index_cpu_to_gpu(res, dev, index)` (`:131-134`) — one suite, two backends, via
`index_factory` keys (`keys_gpu` list at `:96-104`).

**Datasets.** SIFT1M / BIGANN(1bn) / Deep1B from `corpus-texmex.irisa.fr`,
`.fvecs/.ivecs` format (`benchs/README.md:36-80`), stored in sibling subdirs,
scripts self-contained.

**Ground truth computation.** Either shipped (`sift_groundtruth.ivecs`) or, for
data too big for RAM, computed exactly block-by-block:
`contrib/exhaustive_search.py:15` `knn_ground_truth(xq, db_iterator, k, metric)`
— a `faiss.ResultHeap(nq,k)` fed by searching an `IndexFlat` over each DB block
(`index.add(block); D,I=index.search(xq,k); rh.add_result(D,I); index.reset()`),
optionally sharded across GPUs (`:30-39`). `keep_max` flips the heap for
similarity vs distance metrics (`:24`).

**ELIPS verdict.**
- **ADOPT** `evaluate()` verbatim as ELIPS's bench kernel: `(ms/query, recall@k)`
  from `search(xq,k)` vs a gt table. This is the entire Phase 8 QPS/recall
  contract in ~12 lines.
- **ADOPT** the exact-Flat-first sanity gate (R@1==1.0) at the top of every bench
  — it catches a broken loader/metric before you trust approximate numbers.
- **ADOPT** warmup-before-timing (first search pays allocation/JIT; on Metal the
  first dispatch also pays shader/pipeline compilation — warmup is *more*
  important here, not less).
- **ADOPT** `n-recall@1` as the headline metric (`I[:,:n]==gt[:,:1]`) plus
  `knn_intersection_measure` for recall@k with k>1; report ms/query. These are
  the two numbers `05_test_matrix.md`'s recall column wants.
- **ADOPT** `knn_ground_truth` (blockwise `IndexFlat` + result-heap) as ELIPS's
  gt builder — and run it on **Metal** for ELIPS's own datasets (FAISS runs it
  on GPU via `index_cpu_to_all_gpus`; ELIPS's analogue is the Metal Flat index).
  Cache the result to a `.ivecs`-style file so benches are repeatable.
- **ADAPT** the sweep axis: FAISS sweeps `nprobe`/`efSearch`; ELIPS sweeps the
  equivalent per its index types. Keep the `demo_auto_tune` recall-vs-QPS Pareto
  idea even if ELIPS lacks `ParameterSpace` — a manual loop over the tuning
  param producing `(recall, ms/query)` points is enough to draw the frontier.
- **ADAPT datasets**: use SIFT1M (1M×128, has shipped gt) as the primary — it
  fits an M4 and matches FAISS numbers for cross-checking. Add GloVe
  (cosine/IP, tests the metric path) as a second. **SKIP** BIGANN/Deep1B (1bn)
  — no hardware/RAM here and not the point of ELIPS's Phase 8.
- **ADAPT the single-suite-two-backends toggle to ELIPS's three-way axis**: FAISS
  uses `index_factory(key)` + optional `index_cpu_to_gpu`. ELIPS should
  parametrize **one** bench/test body over a backend fixture
  `{CPU, Metal, CUDA(compile-only→skip at runtime)}`, constructing the same
  logical index on each and feeding identical `(xb,xq,gt)`. Mirror the
  `CMAKE_CROSSCOMPILING`/no-hardware guard so the CUDA leg is build-only and the
  Metal+CPU legs actually run on the M4.
- **SKIP** matplotlib plotting in CI (`demo_auto_tune.py:155-172`) — emit the
  `(recall, qps)` points as data (CSV/JSON) and plot offline; the demo already
  degrades gracefully when matplotlib is absent (`:11-19`).

---

## 5. Concrete recommendations for `05_test_matrix.md` & Phase 8

- **Parity tolerances to start with (Metal fp32):** `maxRelErr=6e-3`,
  `pctMaxDiff1=0.10`, `pctMaxDiffN=0.015` (FAISS Flat fp32). Keep a small table
  keyed by (index, dtype); widen only with evidence from the M4, and record the
  final numbers in the matrix. For any dtype<fp32: trust distance rel-err
  (~0.07), drop the rank bound.
- **Oracle:** in-process exact Flat index (CPU for unit tests; Metal Flat for
  big-dataset gt via a blockwise `knn_ground_truth` port, cached to disk).
- **Datasets:** SIFT1M (primary, shipped gt, cross-checks FAISS) + GloVe
  (cosine). Skip billion-scale.
- **Recall metric:** n-recall@1 (`I[:,:n]==gt[:,:1]`) + knn-intersection for k>1;
  report alongside ms/query (QPS = 1000/ms).
- **One suite, three backends:** parametrize a single test/bench body over a
  backend fixture; CPU+Metal run on the M4, CUDA/HIP is compile+link only
  (`CMAKE_CROSSCOMPILING` guard — build it, don't discover/run it).
- **Hygiene to adopt:** warmup before timing; seed + log all RNG; pin OMP
  threads; exact-Flat R@1==1.0 sanity gate before trusting approximate numbers;
  dual assertions (absolute floor + golden-band regression).
