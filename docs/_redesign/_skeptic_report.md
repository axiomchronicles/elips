# Phase 10 — Skeptic / Verifier Report

**Date:** 2026-10-11 · **Branch:** `redesign` (main tree, no worktree, no commit) · **Host:** Apple M4, macOS 26.5.1 (arm64)
**Toolchain:** cmake 4.4.2, ninja 1.13.2, AppleClang, Python 3.14.7 (`~/Library/Python/3.14/bin/pytest`)
**Mandate:** try to prove the redesign's claims wrong. Evidence over assumption. READ/VERIFY only — nothing fixed.

**Headline: 9/9 checks executed. Build PASS, 306 C++ pass / 1 skip / 0 fail, 263 Python pass / 1 skip / 0 fail, docs-snippets 0 fail, ASan subset 46/46 clean, benchmark smoke gate PASS. No CUDA/HIP overclaim found; benchmark numbers match. 0 failures, 0 real overclaims. Worst finding is a LOW wording inconsistency (website calls Metal a "production path"; README/docs call it "Beta").**

---

## Check 1 — Fresh build from documented commands

Command (NEW dir): `cmake -S . -B build-verify -G Ninja -DCMAKE_BUILD_TYPE=Release -DELIPS_ENABLE_PYTHON=ON && cmake --build build-verify -j`

- **Configure: EXIT 0. Build: EXIT 0.** `_core.cpython-314-darwin.so` produced; `elips_tests` linked.
- Warnings: **24, all in tests/benchmarks** — `[[nodiscard]]`-ignored (`BenchGpuSearch.cpp`, `eql_edge_test.cpp`, `MetalBackendTest.cpp`), `erase` deprecated→`remove` (durability/concurrency tests), one `char8_t`→`char32_t` from the googletest header. **Zero warnings in `src/`, `include/`, or the `.mm`/Metal sources.**
- The `MetalBackend.mm:32 unused param` warning noted in the Phase-1 baseline is **gone** (fixed).
- VERDICT: **PASS.**

## Check 2 — Full C++ tests + Metal real-kernel parity actually ran

Command: `ctest --test-dir build-verify`

- **307 tests, 306 passed, 1 skipped, 0 failed, 52.21s.** Skip = `GpuMemoryPoolTest.acquire_and_release` (the single documented baseline skip; unchanged).
- Metal real-kernel parity **RAN, not skipped** (non-trivial wall time proves real GPU work, not an instant `GTEST_SKIP`):
  - #301 `MetalRealKernelParity.flat_distance_and_topk_match_cpu` — **0.69s PASS**
  - #302 `MetalRealKernelParity.k_exceeds_ntotal_pads_sentinel` — 0.01s PASS
  - #303 `MetalBfKnnParity.bf_knn_matches_cpu` — **0.66s PASS**
  - #304 `MetalBfKnnParity.bf_knn_k_exceeds_ntotal_pads_sentinel` — 0.01s PASS
  - #305 `MetalBfKnnParity.gpu_brute_force_index_matches_cpu_indexflat` — **0.29s PASS**
  - #306 `MetalIvfParity.gpu_ivf_matches_cpu_ivfflat` — **1.13s PASS**
- Adversarial source read of `tests/integration/gpu/MetalRealKernelParityTest.cpp`: it obtains a backend via the **production `BackendSelector`** (not `RecordingBackend`), asserts `device_info().backend == "metal"`, runs real `compute_distances_batch` + `top_k`, and only `GTEST_SKIP`s if no Metal device exists. On this M4 it must and did run. The Phase-1 "CAUTION: GPU tests may be mock" concern is **resolved**: real Metal compute is exercised.
- VERDICT: **PASS.** The "Metal validated on Apple M4, real kernels (brute-force + IVF)" claim is backed by tests.

## Check 3 — Python tests

Command: `PYTHONPATH=bindings/python ~/Library/Python/3.14/bin/pytest tests/python -q`

- **263 passed, 1 skipped, 0 failed, 17.8s.** 200 warnings — all `DeprecationWarning` emitted by the rename shims (`write`/`ingest`/`probe`/`probe_text`/`sweep`), i.e. the shims work and warn as designed.
- Skip = `test_native_seam.py:93` (deliberate: "the seam is the one module allowed to import the extension").
- VERDICT: **PASS.**

## Check 4 — Docs snippets

Command: `PYTHONPATH=bindings/python python3 scripts/run_doc_snippets.py docs`

- **77 executed, 25 skipped (no-exec), 0 failed, across 79 files.**
- VERDICT: **PASS.**

## Check 5 — Sanitizer build + subset

Commands: `cmake -S . -B build-asan -G Ninja -DELIPS_SANITIZE=address -DCMAKE_BUILD_TYPE=Debug && cmake --build build-asan -j` → **EXIT 0.** ASan runtime confirmed linked (binary emitted `AddressSanitizer: detect_leaks is not supported on this platform`).

Subset run: `./elips_tests --gtest_filter='*Metal*:*Ivf*:MetalBfKnn*:IndexFlat*:IndexIVFFlat*:IndexFactory*:GpuBackend*:GpuBruteForce*:CpuGpuRecall*'`

- **46 tests / 13 suites ran, 46 passed, EXIT 0, 44.2s. Zero AddressSanitizer errors, zero UBSan `runtime error:`, zero reported leaks.** Covers the real-Metal parity, IVF, bf_knn, IndexFlat/IVFFlat, factory, and GPU-backend paths under ASan.
- **Bounded on purpose:** the full `ctest` under ASan+Debug is dominated by `QuantizedGraphIndex*` recall tests (seconds each in Release → multi-minute under ASan) and exceeded a 5-min window on the first attempt, so I ran the GPU+index subset above instead. **Caveat:** LeakSanitizer is unsupported on macOS/arm64, so *leaks* were not checked — only ASan's heap-overflow / use-after-free / stack-buffer instrumentation (which found nothing) and UBSan.
- VERDICT: **PASS (bounded).**

## Check 6 — Benchmark smoke + sanity gate

Command: `PYTHONPATH=bindings/python python3 benchmarks/bench_ann.py --smoke`

- Ran. `ELIPS IndexFlat (cpu) recall@10 = 1.000` → **exact-Flat sanity gate PASS** (gate asserts exact-Flat recall@k == 1.0, bench_ann.py:233–237). Metal bf_knn recall 1.000 (1-query + batched); IVFFlat recall climbs 0.079→0.384→0.779 with nprobe as expected.
- `faiss-cpu` row: **UNAVAILABLE — no faiss wheel for Python 3.14.7** (environment limitation, not a defect; the committed numbers were generated under Python 3.13 where faiss-cpu 1.15.1 is available).
- NOTE: the smoke run **overwrites `docs/_redesign/_benchmarks.md`** as a side effect; I restored it from git immediately so Check 8's comparison uses the committed file. (`git checkout -- docs/_redesign/_benchmarks.md`.)
- VERDICT: **PASS.**

## Check 7 — Old-name grep (code+docs, excl. `docs/_redesign` + shims)

Pattern: `arena|ingest|place_document|probe_text|seek_hybrid|ElipsInstance|GpuPort|SYCL|Vulkan|sycl|vulkan`

- **SYCL/Vulkan:** zero in `CMakeLists.txt` (no `ELIPS_GPU_SYCL`/Vulkan options — CHANGELOG "Removed" claim verified), zero in real Python source (`bindings/python/elips/`). All doc hits (ADR-GPU-001, adr/README, gpu/index.md, architecture/*, CHANGELOG) explicitly say "dropped / never implemented / removed". **One lingering live code comment:** `include/elips/gpu_engine/GpuBackend.hpp:89` still lists "`CUDA/HIP/Vulkan/SYCL today`" — but ADR-GPU-001 and `docs/gpu/index.md:25` explicitly acknowledge this lingering interface-comment mention. (See Finding F2.)
- **Renamed API terms:** every public hit is a properly marked shim — `GpuPort`/`ElipsInstance`/`place_document`/`seek_hybrid` are `[[deprecated(...)]]` aliases in `elips.hpp`/`GpuBackend.hpp`; Python `Collection.ingest`/`probe_text` carry `.. deprecated::` + `_deprecated(...)`; `engine.py:144 arena = collection` is documented "Prior name for `collection`". Remaining hits are plain-English prose ("a long ingest", "text ingestion"), internal private helpers (`_ingest`/`_probe_text`), the legacy on-disk `WalRecord.arena` field, and the native-seam method names in `_core.pyi` — none are a non-shim public overclaim.
- VERDICT: **PASS** (no non-shim, non-deprecated public hits; two INFO/LOW notes).

## Check 8 — Honesty audit (CUDA/HIP/Metal maturity + benchmark numbers)

- **CUDA/HIP:** every maturity-adjacent mention is qualified. `README.md:94–95` = "Experimental — compiles against the interfaces; not hardware-validated". `docs/gpu/index.md:19–20` = "Experimental — implemented, NOT hardware-validated"; it even volunteers "No cuVS/CAGRA", "FP16/Int8 not implemented", and that `bf_knn` CUDA/HIP are not wired. `CHANGELOG.md:53–54` = "CUDA and HIP/ROCm (experimental, not hardware-validated)". Website `docs.gpu.overview.tsx` = "experimental". **No doc/README/website claims CUDA or HIP is tested / validated / benchmarked / stable.** A targeted grep for CUDA/HIP lines carrying positive-maturity words with the experimental qualifier stripped returned **zero genuine hits** (the single candidate, CHANGELOG:53, was a false positive — its "(experimental, not hardware-validated)" is on the next line).
- **Metal:** README = "Beta"; `docs/gpu/index.md` = "Beta — validated on Apple M4"; both backed by the real-kernel parity tests (Check 2). No claim of Metal validation beyond what the tests cover — **except** the website wording in Finding F1.
- **Benchmark numbers match:** website `docs.benchmarks.tsx` table reproduces `docs/_redesign/_benchmarks.md` **exactly** — IndexFlat 489/1.000/1.981/3.336; bf_knn metal 1-query 24, batched 914; IVFFlat nprobe 1/8/32/128 → 46,614/14,123/6,273/1,616 at recall 0.030/0.152/0.381/0.801; faiss IndexFlatL2 989, faiss IVFFlat nprobe128 2,587/0.793 — plus matching env (Apple M4, Python 3.13, ELIPS 1.1.0, faiss-cpu 1.15.1, dim=128, nb=100000).
- VERDICT: **PASS — no overclaim** (one LOW wording inconsistency, F1).

## Check 9 — Spot-check documented API symbols vs code

Verified against the freshly built module + headers (10+ symbols, Python + C++):

| # | Symbol | Result |
|---|---|---|
| 1 | `elips.Client(dimension=...)` / `PersistentClient` | OK (functional: `Client(dimension=4)` + add/query/count round-trip works) |
| 2 | `elips.index_factory(dim, spec)` | OK (`IndexFactory.hpp:35`) |
| 3 | `elips.IndexFlat` / `IndexIVFFlat` / `IndexIVFPQ` / `IndexIVFScalarQuantizer` / `IndexHNSW` | OK (all present) |
| 4 | `elips.bf_knn` | OK (Python + `BruteForceKnn.hpp:32`) |
| 5 | `elips.has_gpu` (`True`) / `elips.accelerators()` (→ `AcceleratorSpec(name='Apple M4', backend='metal')`) | OK |
| 6 | `Collection.add/upsert/get/update/delete/query/count/peek/modify` | OK (all present) |
| 7 | `Hit.key / .distance / .meta` | OK |
| 8 | `elips.NotTrainedError` / `DimensionMismatchError` | OK (Python + `domain/Errors.hpp:24,43`) |
| 9 | C++ `Database::add_document/search/search_text/hybrid_search` | OK (`elips.hpp:135,149,152,155`, non-deprecated) |
| 10 | C++ `Index::train/is_trained/range_search/type_name` + `IndexFlat`/`IVFFlat` `range_search` | OK (`Index.hpp:18,32,36,54,63`) |

Version SSOT confirmed: `project(... VERSION 1.1.0)` → `ELIPS_VERSION` → `elips.__version__ == 1.1.0`.
- VERDICT: **PASS — 0 missing, 0 wrong signatures.**

---

## Ranked findings (adversarial)

**No failures and no genuine overclaims were found.** The items below are the strongest things a skeptic can hold against the redesign; all are LOW/INFO.

1. **[LOW] Metal maturity wording inconsistency.** Website `elips_docs/src/routes/docs.gpu.overview.tsx:36-37` calls Metal "the implemented, **production** path on Apple silicon", whereas `README.md:93` and `docs/gpu/index.md:18` label it "**Beta**". "production path" reads stronger than "Beta". Not a hard overclaim (CUDA/HIP remain clearly experimental and Metal is backed by real-kernel parity tests), but the two surfaces should use one maturity word. *This is the single most important issue — and it is minor.*
2. **[LOW/INFO] Stale dropped-backend mention in a live code comment.** `include/elips/gpu_engine/GpuBackend.hpp:89` lists "CUDA/HIP/Vulkan/SYCL today" among backends lacking an IVF-scan kernel; Vulkan/SYCL are removed. ADR-GPU-001 and `docs/gpu/index.md:25` already acknowledge this lingering comment, so it is disclosed, not hidden.
3. **[INFO] Untracked stale build dir leaks old options.** `bindings/python/build/` (gitignored, `git ls-files` = 0) still contains a `CMakeCache.txt` with `ELIPS_GPU_SYCL:BOOL=OFF`/Vulkan and an `accelerator.py` copy mentioning `sycl`/`vulkan`. Not part of the repo; could confuse a grep. Harmless.
4. **[INFO] Benchmark reproducibility is Python-version-sensitive.** Under Python 3.14 the faiss comparison row is UNAVAILABLE (no 3.14 wheel); the committed numbers require Python 3.13 + faiss-cpu 1.15.1. The docs state the environment, so this is a reproducibility caveat, not a discrepancy.
5. **[INFO] Side effect of the smoke command.** `bench_ann.py --smoke` rewrites `docs/_redesign/_benchmarks.md`; restored from git during this audit. Running the documented smoke dirties a tracked file — worth a `--no-write` guard, out of scope here.

### Pre-existing uncommitted tree state (not caused by this audit)
`git status` shows `M THIRD_PARTY_NOTICES.md` and untracked `docs/_redesign/final_report.md` (mtimes 00:54–00:55, predate this session) left by an earlier Phase 9/10 step, plus `Testing/` and `tmp/`. Left untouched (read/verify mandate). This report adds only `docs/_redesign/_skeptic_report.md`.
