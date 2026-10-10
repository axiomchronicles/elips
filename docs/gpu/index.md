# GPU Acceleration Guide

ELIPS is CPU-first and GPU-optional. The GPU layer accelerates distance
computation, top-k selection, IVF scan, and standalone brute-force k-NN. Domain
code depends only on the abstract `GpuBackend` interface; concrete backends are
compiled in behind flags.

This guide is the authoritative source for **what actually works on which
hardware**. Where older design docs or ADRs over-claimed (cuVS/CAGRA, a 4th/5th
backend, silent fallback), this page and the corrected [GPU ADRs](../adr/README.md)
win.

## Backend maturity

| Backend | Status | What is real | Validation |
|---|---|---|---|
| **CPU** | **Stable** | `IndexFlat`, `IndexHNSW`, `IndexIVFFlat/IVFPQ`, exact `bf_knn` | Full CTest + pytest suite, recall oracle |
| **Metal** (Apple) | **Beta — validated on Apple M4** | GPU distance (tiled GEMM), **GPU top-k**, **GPU IVF scan** (2-pass), `bf_knn(device="metal")`, unified-memory zero-copy | Real-kernel parity tests vs CPU oracle (exact label + tight distance tolerance) |
| **CUDA** (NVIDIA) | **Experimental — implemented, NOT hardware-validated** | cuBLAS GEMM + custom distance/top-k kernels; same `GpuBackend` interface | Builds from source; container compile-check in x86_64 CI only. No GPU hardware run. |
| **HIP** (AMD ROCm) | **Experimental — implemented, NOT hardware-validated** | rocBLAS + custom kernels; same interface | Builds from source; container compile-check only. No GPU hardware run. |

Notes that matter:

- **There is no SYCL or Vulkan backend.** Earlier plans mentioned them; they were
  never implemented and have been dropped. Lingering `Vulkan/SYCL` mentions in
  code comments are placeholders.
- **No cuVS / CAGRA.** The CUDA backend uses cuBLAS + hand-written kernels only.
  The `CagraGraph` algorithm enum and `supports_cagra` flag are aspirational
  scaffolding, not a validated path (see [ADR-GPU-002](../adr/ADR-GPU-002-cuvs-vs-custom-kernels.md)).
- **FP16/Int8 compute is not implemented.** All GPU math runs in FP32; on Metal
  FP16 is storage-only. `enable_fp16_search` / non-FP32 `search_precision` are
  effectively no-ops today (see [ADR-GPU-007](../adr/ADR-GPU-007-precision-strategy.md)).

## Selecting a device

The `device=` string chooses a backend (case-insensitive):

| `device=` | Meaning |
|---|---|
| `"cpu"` | CPU only; never touches the GPU |
| `"auto"` | Use a GPU if one is present, else CPU (CPU fallback always permitted, and logged if it happens) |
| `"metal"` | Require the Metal backend |
| `"cuda"` / `"cuda:N"` | Require the CUDA backend (any device / device index N) |
| `"hip"` / `"hip:N"` | Require the HIP backend (any device / device index N) |

```python
import elips

# Validated path on Apple Silicon:
db = elips.connect(":memory:", dimension=128, metric="cosine", device="metal")

# Portable: GPU if available, otherwise CPU (demotion is logged, not silent):
db = elips.connect(":memory:", dimension=128, metric="cosine", device="auto")
```
<!-- APPEND-MARKER -->

## No-silent-fallback

A **named** backend that is absent is an error, not a quiet downgrade:

- `device="metal"` (or `cuda`/`hip`) when that backend is unavailable raises
  **`BackendUnavailableError`** at open time.
- Pass **`fallback=True`** to permit a demotion to CPU instead. The demotion is
  **logged** to stderr:
  `[elips] warning: GPU backend 'metal' unavailable; falling back to CPU`.
- `device="auto"` always permits CPU fallback (CPU is always present), and still
  logs when it had to demote.
- At query time, a backend error (OOM, kernel launch) degrades to the CPU path
  and increments `GpuMetricsSnapshot.fallback_events_total`, so a silent
  performance cliff is visible in metrics.

```python no-exec
# Hard-require Metal (raises BackendUnavailableError on a non-Apple box):
db = elips.connect(path, dimension=128, device="metal")           # fallback=False (default)

# Prefer Metal, accept a logged CPU demotion:
db = elips.connect(path, dimension=128, device="metal", fallback=True)
```

Legacy note: `GpuPolicy::RequireGpu` is not a per-query GPU guarantee. Prefer a
named `device=` with `fallback=False` to fail hard when the GPU is missing.

## Standalone brute-force k-NN (`bf_knn`)

`bf_knn` runs an exact k-NN over raw arrays with no index build. It is the
simplest way to exercise the GPU and the clearest Metal win when batched.

```python
import numpy as np, elips

xq = np.random.rand(1000, 128).astype("float32")   # queries
xb = np.random.rand(100000, 128).astype("float32") # database
D, I = elips.bf_knn(xq, xb, k=10, metric="l2", device="metal")
# D: (nq, k) float32 distances, I: (nq, k) int64 row-index labels.
# k > nb pads labels with -1 and distances with +inf (FAISS heap-padding contract).
```

- `device` for `bf_knn` is `"auto"`, `"cpu"`, or `"metal"` (CUDA/HIP are not wired
  into `bf_knn`, matching their experimental status).
- Results match the CPU oracle exactly: both select the k smallest distances with
  a `(distance, row-index)` ascending tiebreak.
- Cosine assumes L2-normalized inputs (ELIPS convention).
- **Batch your queries.** One batched call amortizes launch + sync; on Apple M4,
  batched `bf_knn` reaches ~914 QPS vs ~24 QPS single-query (see
  [Which index?](../overview/choosing-an-index.md)).

## Metal internals (Apple Silicon)

The Metal path runs real GPU kernels loaded at runtime from
`elips_kernels.metallib` (built by the `xcrun metal → .air → metallib` chain,
discovered via `ELIPS_METALLIB_PATH` env → compiled-in build path → next to the
dylib):

- `elips_l2_matrix` / `elips_ip_matrix` — tiled-GEMM distance matrices (no MPS dep)
- `topk_select` — on-GPU top-k (replaces the old CPU `std::partial_sort`)
- `elips_ivf_scan_list` + `elips_ivf_merge_lists` — two-pass GPU IVF scan

**Unified memory:** every Metal buffer is `MTLResourceStorageModeShared`, so CPU
and GPU share one allocation. "Upload"/"download" is a `std::memcpy` over
`[buffer contents]` — there is no PCIe blit and no staging buffer. Practical
consequences:

- Transfers are nearly free; the cost is compute, not copy.
- Larger-than-"VRAM" datasets are practical (one shared pool, no device-only cap).
- The CPU and GPU compete for the same memory bandwidth, so the GPU win over
  NEON-optimized CPU is real but smaller than on a discrete GPU — it shows up in
  **batched** throughput, not single-query latency.

## Building the experimental backends

CUDA and HIP build from source and are compile-checked in x86_64 CI containers;
they are **not run on GPU hardware** and must not be cited as validated.

```bash
# CUDA (needs CUDA toolkit; cuBLAS):
cmake -B build -DELIPS_ENABLE_GPU=ON -DELIPS_ENABLE_CUDA=ON
cmake --build build

# HIP / ROCm (needs ROCm; rocBLAS):
cmake -B build -DELIPS_ENABLE_GPU=ON -DELIPS_ENABLE_ROCM=ON
cmake --build build

# Metal (Apple Silicon; auto-on on Darwin arm64):
cmake -B build -DELIPS_ENABLE_GPU=ON -DELIPS_ENABLE_METAL=ON
cmake --build build
```

The repository ships container compile scripts for the x86_64 CI compile-check of
CUDA/HIP; see the build-system docs and CI config.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `BackendUnavailableError` at open | Named `device=` backend not present | Use `device="auto"`, or pass `fallback=True`, or install the backend |
| `[elips] warning: GPU backend ... falling back to CPU` | Requested GPU absent, `fallback=True` | Expected; the engine ran on CPU. Remove `fallback` to hard-fail instead |
| `elips_kernels.metallib not found` (NSLog) | metallib not built/installed | Rebuild with Metal enabled, or set `ELIPS_METALLIB_PATH` to the built `.metallib` |
| GPU slower than CPU for one query | Launch + sync overhead dominates small work | Batch queries (`search(xq, k)` over all queries, or batched `bf_knn`) |
| `accelerators()` returns `[]` | CPU-only build, or no GPU present | Confirm the GPU bindings were compiled (`elips.__init__` degrades to empty list otherwise) |
| FP16 flag seems ignored | FP16 compute not implemented | Expected — all GPU math is FP32 today |

## Observability

- `elips.accelerators()` → list of detected devices (`AcceleratorSpec`: name,
  backend, memory, `unified_memory`, `supports_fp16`). Empty on CPU-only.
- `GpuDeviceInfo` / `gpu_device_info()` → active device or the synthetic CPU
  fallback info.
- `GpuMetricsSnapshot.fallback_events_total` → count of query-time CPU demotions.

## See also

- [Which index should I use?](../overview/choosing-an-index.md) — measured CPU vs Metal vs faiss-cpu
- [Architecture](../architecture.md) — the GPU layer in context
- [GPU ADRs](../adr/README.md) — corrected design decisions
