# Phase 9C — Docs verification (Architecture / Overview / ADR + GPU & index guides)

Date: 2026-10-11. Owner scope: `docs/architecture/**`, `docs/architecture.md`,
`docs/overview/**`, `docs/adr/**`, new `docs/gpu/`, and `docs/cpp/performance/benchmarking.md`.
No code / other-docs / website / git touched.

## Ground truth used
- `docs/_redesign/{03_naming_map, 04_architecture, 02_gap_analysis, _faiss_3I_metal, _benchmarks}.md`
- Real code: `include/elips/gpu_engine/{GpuBackend,BackendSelector,GpuDeviceManager,GpuConfig,BruteForceKnn}.hpp`,
  `src/gpu_engine/{BackendSelector,GpuDeviceManager}.cpp`,
  `src/gpu_engine/backends/{metal/MetalBackend.mm, metal/shaders/elips_distance_kernels.metal, cuda/CudaBackend.cu, hip/HipBackend.hip}`,
  `src/Database.cpp` (fallback logging), `bindings/python/elips/config/connect.py`,
  `bindings/python/elips/_core.pyi` (`bf_knn`), `CMakeLists.txt` (options),
  `tests/integration/gpu/Metal*ParityTest.cpp`, `tests/parity/CpuGpuRecallParityTest.cpp`.

## Verified facts (code-confirmed)
- Interface is `GpuBackend` (`GpuPort` = deprecated alias); selector `BackendSelector`
  (`GpuSelector` alias); batcher `QueryBatcher`. Core base `Index`; `IndexFlat`/`IndexHNSW`/IVF family.
- Device string grammar (`connect.py:_build_gpu_config`): `cpu`/`auto`/`metal`/`cuda[:N]`/`hip[:N]`.
- No-silent-fallback: `Database.cpp` throws `BackendUnavailableError` unless `fallback`/Auto,
  then prints `[elips] warning: GPU backend ... falling back to CPU` to stderr.
- Metal real kernels loaded from `elips_kernels.metallib`: `elips_l2_matrix`, `elips_ip_matrix`,
  `topk_select` (GPU top-k), `elips_ivf_scan_list` + `elips_ivf_merge_lists` (2-pass IVF). Unified
  `MTLResourceStorageModeShared`. Parity tests exist (Metal real-kernel / IVF / bf_knn).
- CUDA backend = cuBLAS GEMM + custom kernels (`CudaBackend.cu`, header "EXPERIMENTAL / NOT
  HARDWARE-VALIDATED"); **no cuVS, no CAGRA code path**. HIP = rocBLAS + custom kernels, experimental.
- `bf_knn` device is `auto|cpu|metal` only (not cuda/hip) — `_core.pyi:1783`.
- FP16/Int8 compute not implemented; `enable_fp16_search` default false; Metal fp16 storage-only.
- CMake canonical options `ELIPS_ENABLE_METAL/CUDA/ROCM/PYTHON/C_API/BLAS`, `BUILD_TESTING`,
  `ELIPS_OPT_LEVEL`; old `ELIPS_GPU_*`/`ELIPS_BUILD_*` kept as aliases.
- Benchmarks (`_benchmarks.md`): measured on Apple M4; numbers transcribed verbatim into the index guide.

## Files updated
- `docs/architecture.md` — full rewrite: Client/Database, Collection-on-Index, Index API, GPU layer diagram + prose.
- `docs/overview/architecture.md` — full rewrite: new runtime-shape diagram, Index API, GPU flow, no-silent-fallback.
- `docs/overview/introduction.md` — mental model + example to `add`/`search`/`collection`/`n_results`; index-guide link.
- `docs/architecture/system-overview.md` — C4 container diagram, 5-tier ASCII, layer descriptions (API/Query/Index/GPU), design-decision + cross-cutting names.
- `docs/adr/README.md` — added GPU ADR table with honest statuses.
- `docs/adr/ADR-GPU-001..008` — corrected: 001 amended (GpuBackend, SYCL/Vulkan dropped, maturity);
  002 **Superseded** (no cuVS/CAGRA — cuBLAS+custom only); 003 **Superseded** (CAGRA export never built);
  004/005/006/007/008 amended (pool reality, QueryBatcher rename, M4-validated unified memory,
  FP16/Int8 unimplemented, **logged no-silent-fallback** replacing the "silent/transparent" claim).
- `docs/gpu/index.md` — **new** GPU guide: maturity table, device selection, no-silent-fallback,
  `bf_knn`, Metal unified-memory internals, build steps, troubleshooting, observability.
- `docs/overview/choosing-an-index.md` — **new** "Which index?" guide with the real M4 benchmark table
  (honest: faiss faster on exact CPU + IVF build; ELIPS batched Metal `bf_knn` bright spot).
- `docs/cpp/performance/benchmarking.md` — added Apple M4 hardware/versions note + index-guide link (already pointed at `bench_ann.py`).
- `docs/architecture/faiss-gpu-acceleration-design.md`, `gpu-device-reporting-audit.md`,
  `gpu-runtime-stability-audit.md`, `runtime-flow.md`, `memory-flow.md`, `binding-flow.md`,
  `build-system.md`, `overview/design-principles.md` — correction/rename banners (historical/internal
  docs; narratives accurate, names mapped to current, pointers to authoritative GPU guide/ADRs).
- `docs/architecture/local-embedding-system.md` — inline rename of `place`/`seek`/`vault` examples to `add`/`search`/`collection`.

## Maturity table (authoritative — mirrors docs/gpu/index.md)

| Backend | Status | What is real | Validation |
|---|---|---|---|
| CPU | Stable | IndexFlat, IndexHNSW, IVFFlat/IVFPQ, exact bf_knn | Full CTest+pytest, recall oracle |
| Metal | Beta — validated on Apple M4 | GPU distance (tiled GEMM), GPU top-k, GPU IVF scan (2-pass), bf_knn(device="metal"), unified-memory zero-copy | Real-kernel parity tests vs CPU oracle |
| CUDA | Experimental — implemented, NOT hw-validated | cuBLAS GEMM + custom distance/top-k kernels | Build-from-source; x86_64 CI container compile-check only |
| HIP | Experimental — implemented, NOT hw-validated | rocBLAS + custom kernels | Build-from-source; container compile-check only |

## Remaining gaps / not done (honest)
- `project-history.md` left as-is (historical record; old names are legitimate history).
- Deep internal flow docs (`runtime-flow`, `memory-flow`, `binding-flow`, `build-system`,
  `design-principles`) got rename banners rather than full line-by-line rewrites — proportionate and
  avoids introducing unverified claims; their flow/mechanics narratives were not re-audited line by line.
- `faiss-gpu-acceleration-design.md` carries a "historical design analysis (superseded in parts)"
  banner rather than a 30 KB rewrite; it still references retired `IndexSnapshot` and stale
  `/Users/kuroyami/elips/` paths in the body.
- Code-level selector ranking still numerically favors cuda(100)/hip(90) over metal(80)
  (`BackendSelector.cpp`) — only matters in a hypothetical multi-backend-present build; documented as
  a known gap, not fixed (out of docs scope).
- `system-overview.md` error-class names (`DimensionMismatch` etc.) not aligned to the `*Error`
  suffix map — minor, left (old names remain valid aliases).
- Did not touch docs/python, docs/cpp (except benchmarking.md), docs/internals, docs/eql, website — per scope.
