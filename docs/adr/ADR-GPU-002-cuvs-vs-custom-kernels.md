# ADR-GPU-002: cuVS vs Custom CUDA Kernels

**Status:** Superseded (2026-10-11) — the cuVS path was never implemented
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

For the CUDA backend, we considered whether to depend on NVIDIA cuVS (CAGRA,
IVF-PQ, etc.) or write our own CUDA kernels for distance computation.

## Original Decision (not realized)

The original plan was "both": cuVS as the primary path (CAGRA build, IVF-PQ
compression) with hand-written kernels as a fallback, claiming an 8–12x CAGRA
build speedup.

## Reality Update (2026-10-11) — this ADR is superseded

**cuVS / CAGRA was never integrated.** The shipped CUDA backend
(`src/gpu_engine/backends/cuda/CudaBackend.cu`, header-labelled *EXPERIMENTAL /
NOT HARDWARE-VALIDATED*) uses only:

- **cuBLAS GEMM** (`CUDA::cublas`) to form batch distance matrices
- **custom hand-written CUDA kernels** (`kernels/elips_cuda_kernels.cu`) for the
  distance finalize + top-k selection

There is **no `find_package(cuvs)`** and **no CAGRA code path**. The
`GpuIndexAlgorithm::CagraGraph` enumerator and the `supports_cagra` device flag
still exist, but on CUDA they are an **unvalidated, aspirational** label, not a
cuVS-backed implementation. The 8–12x build-speedup claim is **unsubstantiated**
and should not be cited.

Because the CUDA backend is experimental and not hardware-validated, no cuVS
integration or benchmark claims are made. If cuVS is ever adopted, it will get a
fresh ADR superseding this one.

## Consequences

- CUDA backend links `CUDA::cudart` + `CUDA::cublas` only; no cuVS dependency.
- CAGRA on CUDA is **not available**; treat `CagraGraph` as experimental scaffolding.
- Any "CAGRA 8–12x" statement in older docs is wrong — see the
  [GPU guide](../gpu/index.md) for the real maturity table.
