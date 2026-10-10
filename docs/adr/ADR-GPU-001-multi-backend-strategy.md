# ADR-GPU-001: Multi-Backend Abstraction Strategy

**Status:** Accepted (amended 2026-10-11 — see Reality Update)
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

ELIPS needs GPU acceleration across Apple Metal, NVIDIA CUDA, and AMD ROCm/HIP.
Each backend has distinct APIs, memory models, and compilation requirements.

## Decision

We use a single abstract interface `GpuBackend` that all backends implement.
Domain code depends only on `GpuBackend`, never on concrete backend types.
Backend selection occurs at startup via `GpuDeviceManager` and `BackendSelector`.

```
GpuBackend (abstract interface)
  <- MetalBackend (MSL compute kernels)   — validated on Apple M4
  <- CudaBackend  (cuBLAS + custom kernels) — EXPERIMENTAL, not hw-validated
  <- HipBackend   (rocBLAS + custom kernels) — EXPERIMENTAL, not hw-validated
```

## Rationale

- **Single dispatch point**: Domain code calls `backend->compute_distances_batch()` regardless of vendor
- **Compile-time safety**: Backend headers are never included in domain code (guarded by `ELIPS_GPU_ENABLED`)
- **Runtime flexibility**: Backend can be swapped without recompiling domain code
- **Testing simplicity**: the Metal path is validated against the CPU oracle by real-kernel parity tests

## Reality Update (2026-10-11)

- The interface is named **`GpuBackend`** (was `GpuPort`); the selector is
  **`BackendSelector`** (was `GpuSelector`); the per-index base is **`GpuIndex`**
  (was `GpuIndexPort`), extending the core **`Index`** (was `IndexPort`).
- **Only three backends exist: Metal, CUDA, HIP.** Earlier planning notes floated
  SYCL and Vulkan backends; **those were never implemented and have been dropped**
  from the design. The lingering `Vulkan/SYCL` mentions in interface comments are
  placeholders, not shipped backends.
- **Maturity is not uniform.** Metal is hardware-validated on Apple M4 (real
  distance/top-k/IVF-scan kernels + parity tests). CUDA and HIP are implemented
  against the same interface but are **EXPERIMENTAL and not hardware-validated**;
  they build from source / compile-check only. See the [GPU guide](../gpu/index.md).
- CMake flags are `ELIPS_ENABLE_METAL` / `ELIPS_ENABLE_CUDA` / `ELIPS_ENABLE_ROCM`
  under the `ELIPS_ENABLE_GPU` umbrella (renamed from `ELIPS_GPU_*`).

## Consequences

- Each backend must implement every `GpuBackend` method; some (e.g. `ivf_scan`)
  default to `BackendUnavailable` on backends without a real kernel.
- `GpuIndex` extends `Index`, keeping the existing DI seam intact.
