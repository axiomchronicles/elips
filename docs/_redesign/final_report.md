# ELIPS Redesign — Final Report (Phase 10)

**Date:** 2026-10-11 · **Branch:** `redesign` · **Version at report time:** 1.1.0 (SSOT across CMake / `elips.__version__` / setup.py / pyproject)

This is the project's integrity statement. It is written to be scrupulously honest about
what was built, what was tested, on what hardware, and what remains. Where a claim could be
read as stronger than the evidence, the weaker, true reading is the one stated.

---

## 1. What was done, phase by phase

| Phase | Title | Outcome |
|---|---|---|
| 1 | Inventory & baseline | True status captured. **Premise corrected:** the CPU core was already mature (HNSW/IVF-quant/EQL/WAL/metadata real and tested), not immature. Immaturity was concentrated in GPU backends, naming, and docs truth. Baseline green: build 0, ctest 255/256 (1 skip), pytest 239/244. |
| 2 | Prune GPU backends | Deleted SYCL and Vulkan entirely (non-compiling `#error` stubs): source dirs, CMake wiring, device probe/rank, 2 ADRs, 16 doc + 3 website files. Backend set now Metal / CUDA / HIP / CPU. Repo-wide `sycl\|vulkan\|spirv` grep clean outside `docs/_redesign`. |
| 3 | Study FAISS | Cloned `facebookresearch/faiss` v1.15.1. **Major correction (ADR-R1):** upstream now ships a real `faiss/gpu_metal/` backend (~7700 LOC incl. a real on-GPU `topk_threadgroup_2048`). Phase 7 Metal became *adapt a precedent*, not greenfield. |
| 4 | Gap analysis | 7 blockers registered, each with an owner phase: vocabulary chaos, no `index_factory`, no `Client`/`PersistentClient`, GPU top-k on CPU, naive GPU distance kernel, no real-GPU test, docs actively wrong. Biggest core gap: no CPU IVF (was GPU-only). |
| 5 | Naming & public API redesign | Deep rename (ADR-R2, user's choice) across C++ / Python / EQL / CMake in 8 verified batches — `Vault`→`Collection`, `ElipsInstance`→`Database`, `IndexPort`→`Index`, `ExactIndex`→`IndexFlat`, `place`→`add`, `seek`→`search`, etc. Added Chroma `Client()`/`PersistentClient(path=)` + Collection methods. Deprecation shims for every old public name. On-disk format untouched (persistence tests green). |
| 6 | Core engine maturity (CPU) | CPU IVF family added (`IndexIVFFlat`/`IndexIVFPQ`/`IndexIVFScalarQuantizer`) over a FAISS-style `index_factory` grammar; FAISS Index API exposed in Python (numpy I/O); IDMap for arbitrary Chroma string ids; `range_search` on Flat+IVFFlat; typed errors; k>ntotal pads -1. Recall property-tested vs the IndexFlat brute-force oracle. |
| 7 | GPU acceleration (Metal first) | **Metal validated on Apple M4:** real GPU tiled-GEMM distance, real on-GPU top-k (replacing CPU `std::partial_sort`), GPU IVF scan, standalone `bf_knn` — all with real-kernel parity tests that RAN+PASSED on the M4. Device selection with no silent fallback. CUDA + HIP backends implemented, gated OFF, labelled EXPERIMENTAL (see §2 for the honest validation status). |
| 8 | Build / packaging / test infra | FAISS-style CMake (`ELIPS_ENABLE_*`, `ELIPS_OPT_LEVEL` with real SIMD flags, `ELIPS_ENABLE_C_API`, `install(EXPORT)` + `find_package(elips)` with a working consumer smoke test); numpy runtime dep + version SSOT 1.1.0; CI matrix incl. EXPERIMENTAL cuda/hip compile jobs, sanitizers, docs-snippet runner, GPU memory-cycle tests. Latent CMake breaks fixed; 0 link warnings. |
| 9 | Documentation | Docs truth pass: Python/C++/architecture/GPU/EQL/website prose reconciled with shipped code (the website previously *denied* the shipped `add/search/get/delete` surface; 10 GPU ADRs previously overclaimed unimplemented backends). |
| 10 | Release readiness | This report, THIRD_PARTY_NOTICES verification, CHANGELOG/MIGRATION finalization. |

---

## 2. What is tested, and on what hardware

**This is the most important section. Read the hardware column, not just the checkmark.**

| Backend | Implementation | Validated on real hardware? | Evidence |
|---|---|---|---|
| **CPU** | complete | **YES** — full `ctest` + `pytest`, recall vs exact oracle | 307 ctest / ~263 pytest on this machine; recall@10 recorded in `05_test_matrix.md` |
| **Metal** | complete | **YES — Apple M4 (arm64)** | Real-kernel parity tests RAN+PASSED on the M4: `MetalRealKernelParity` (L2/IP/cosine × dims{2,64,128,384} × n{1,1e3,1e4} × k{1,10,100} + k>ntotal, tol 6e-3), `MetalIvfParity` (recall 1.0 cosine/dot/L2-d32, 0.994–0.997 L2-d128 vs CPU IVFFlat; exact at nprobe=nlist), 3 `bf_knn` parity tests |
| **CUDA** | complete, EXPERIMENTAL, gated OFF | **NO — never hardware-validated** | cuBLAS SGEMM distance + on-device k-select (adapted from FAISS block/warp-select). Container-compile script + `Dockerfile.cuda` exist for x86_64 CI, but locally `nvcc`'s `cicc` **segfaults under QEMU amd64-on-arm64 emulation** (a toolchain limitation, crashes before our sources). Validated here only by host-clang syntax checks + exact CI command recorded. |
| **HIP / ROCm** | complete, EXPERIMENTAL, gated OFF | **NO — never hardware-validated** | hipBLAS SGEMM + device top-k. `hipcc` compiled the backend **once under emulation** (pre-merge interface); the re-run against the merged interface was blocked because **Docker was reset after the image pulls filled the disk**, so it was validated via host-clang `-fsyntax-only` + a `static_assert(!is_abstract_v<HipBackend>)` against the merged `GpuBackend.hpp`. |

**Container-compile reality, stated plainly:** CUDA and HIP are CI-only on x86_64. On this
development machine (Apple Silicon) neither backend's compiler runs to completion: the CUDA
toolchain segfaults under QEMU emulation, and HIP compiled once under emulation before a Docker
reset (triggered by the image pulls exhausting disk) removed the environment. Both backends are
fully implemented, gated OFF by default, labelled EXPERIMENTAL in headers / CMake / device names,
and carry hardware-gated tests and compile scripts — but **no CUDA or HIP code has ever executed
on a real GPU.** Do not represent them otherwise.

**Test counts (this machine, default CPU+Metal build):**
- `ctest`: **307** (306 + 1 skip)
- `pytest`: **~263**
- Doc-snippet runner (`scripts/run_doc_snippets.py`): **0 failed**

---

## 3. Benchmark headline

Measured on Apple M4 (arm64), nb=100k, d=128, nq=1k, k=10, L2, single-query unless noted. Full
table in `_benchmarks.md`. Reported as measured — wins and losses both.

- **faiss-cpu is faster on exact CPU search:** faiss `IndexFlatL2` 989 QPS vs ELIPS `IndexFlat` 489 QPS (~2×).
- **faiss-cpu builds IVF far faster:** ~0.25 s vs ELIPS 25–71 s. ELIPS IVF *build time* is a clear
  optimization target; recall matches within noise (both ~0.80 @ nprobe=128).
- **ELIPS bright spot — Metal batched `bf_knn`: 914 QPS @ recall 1.000** (amortized over one batched
  call), competitive with faiss-cpu exact. Single-query Metal is dispatch-bound (24 QPS) — batch it.
- Recall is correct everywhere: exact = 1.000; ELIPS and faiss IVF recall track each other.

Honest read: ELIPS is **not** faster than FAISS on CPU today, and its IVF build is slow. Its
genuine win is the validated Apple-GPU batched brute-force path.

---

## 4. Known limitations / honest gaps

**GPU**
- CUDA and HIP are **not hardware-validated** (see §2). EXPERIMENTAL, gated OFF.
- **No GPU IVFPQ-on-device.** The Metal IVF path is an IVF**Flat** scan (`ivf_scan_list` +
  `ivf_merge_lists`); PQ codes are not scanned on-device.
- Single-query Metal dispatch is latency-bound; the GPU win requires batching.

**Python / API surface**
- `Collection.update(ids=…)` is not yet string-id-aware (the IDMap covers add/get/delete/upsert/query).
- `where_document` raises `NotImplementedError`.
- `delete_collection` / `reset` raise `NotImplementedError`.
- Python does **not** expose `range_search` / `reconstruct` / `read_index` / `write_index` /
  GPU-transfer helpers (`index_cpu_to_gpu`, `StandardGpuResources`, `get_num_gpus`). C++ has
  `range_search` on Flat+IVFFlat only.
- `Client()` is **dimension-first** — it does not infer dimension from a default embedder like Chroma.

**Semantics**
- The `euclidean` metric returns **true √L2 distance, not FAISS squared-L2.** Ranking is identical;
  absolute distance values differ. Documented; no squared-mode kernel yet.

**Engine / perf**
- Search is **single-threaded** — no parallel-search thread pool (deferred until benchmarked).
- IVF build is slow vs FAISS (see §3).
- `IndexRefine` / `refine_fraction` not implemented (niche; the knob exists but is inert).

**Build / packaging**
- No abi3 wheel (per-interpreter wheels only); FAISS-style `dd` dispatch deferred.
- CUDA/HIP kernel-file naming cleanup still tracked.
- The website still needs a CI `vite` build step.
- Pre-existing `ruff` lint debt not yet cleared.

---

## 5. Remaining roadmap (prioritized, honest)

1. **Hardware-validate CUDA on a real NVIDIA GPU** (x86_64 CI with a GPU runner) — the single
   biggest credibility gap. Then HIP on a real AMD GPU.
2. **Optimize CPU IVF build time** (25–71 s → target sub-second class) to close the FAISS gap.
3. **GPU IVFPQ-on-device** (scan PQ codes on Metal, then CUDA/HIP) — the next real GPU feature.
4. Finish the Python Index API surface: `range_search`/`reconstruct`/`read_index`/`write_index`,
   GPU-transfer helpers.
5. `Collection.update` string-id awareness; implement `where_document`, `delete_collection`, `reset`.
6. Parallel-search thread pool (after a benchmark justifies it).
7. `IndexRefine`/`refine_fraction`; optional squared-L2 metric mode.
8. Packaging polish: abi3 wheel, `dd` dispatch, website CI build, ruff pass.

---

## 6. Deprecation & versioning

**Deprecation shims:** every old public name still works this release — Python emits
`DeprecationWarning`, C++ is marked `[[deprecated]]`, EQL accepts old verbs as parse-time synonyms,
CMake emits `message(DEPRECATION)`. **All old names are scheduled for removal in the next minor
release.** Full old→new tables are in `MIGRATION.md`. The on-disk format is unchanged — existing
databases open without migration.

**SemVer recommendation — MAJOR bump (recommend 2.0.0):** this release renames the entire public
API across C++/Python/EQL/CMake and **removes** the SYCL and Vulkan backends. Even though old names
still resolve via shims, removing backends and re-architecting the public surface is a breaking
change under SemVer, and the shims themselves are slated for removal next version. A `2.0.0` release
communicates that honestly to downstream users. The SSOT is currently **1.1.0** across all manifests
— **the actual version bump is left to the maintainer**; this report only records the recommendation.

---

*Prepared as the Phase-10 deliverable. The posture throughout: reality wins (ground rule #7), and
numbers and validation status are reported as measured (ground rule #1).*
