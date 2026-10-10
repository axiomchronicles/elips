# ADR-GPU-001: Multi-Backend Abstraction Strategy

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

ELIPS needs GPU acceleration across Apple Metal, NVIDIA CUDA, and AMD ROCm/HIP. Each backend has distinct APIs, memory models, and compilation requirements. Metal is the hardware-validated backend; CUDA and HIP are experimental and not yet hardware-validated.

## Decision

We use a single abstract interface `GpuPort` that all backends implement. Domain code depends only on `GpuPort`, never on concrete backend types. Backend selection occurs at startup via `GpuDeviceManager` and `GpuSelector`.

```
GpuPort (abstract interface)
  <- MetalBackend (MPSGraph, MSL compute kernels)
  <- CudaBackend  (cuVS/CAGRA, cuBLAS)
  <- HipBackend   (hipVS/CAGRA, rocBLAS)
```

## Rationale

- **Single dispatch point**: Domain code calls `backend->compute_distances_batch()` regardless of vendor
- **Compile-time safety**: Backend headers are never included in domain code (`elips.hpp` uses `#ifdef ELIPS_GPU_ENABLED` guards)
- **Runtime flexibility**: Backend can be swapped without recompiling domain code
- **Testing simplicity**: Mock GPU backends implement the same interface

## Consequences

- Each backend must implement every `GpuPort` method, even if some are stubs
- `GpuIndexPort` extends `IndexPort`, keeping the existing DI seam intact
- Build system uses per-backend CMake flags (`ELIPS_GPU_CUDA`, etc.)