# ADR-GPU-009: SYCL Portability vs Performance

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

SYCL provides cross-platform GPU compute that can target Intel Xe GPUs, and (via adaptiveCpp/hipSYCL) also NVIDIA and AMD GPUs. This creates a choice: use SYCL as the primary portable path, or keep it as an additional backend alongside native CUDA/HIP/Metal.

## Decision

SYCL is an additional backend, not the primary path. Native backends (CUDA, HIP, Metal) are preferred when available. SYCL serves as a secondary path for Intel GPUs and a portability option.

Priority order:
1. CUDA (NVIDIA) — best hardware, cuVS/CAGRA, most mature
2. HIP (AMD) — hipVS/CAGRA, API-compatible with CUDA
3. Metal (Apple) — zero-copy unified memory, MPSGraph
4. SYCL (Intel) — Intel Arc/Flex/Max, oneMKL
5. Vulkan (universal) — any GPU with Vulkan 1.2+

## Rationale

- Native backends provide vendor-specific optimizations (cuVS, hipVS, MPS)
- SYCL adds compilation complexity (Intel DPC++ or adaptiveCpp)
- SYCL's oneMKL GEMM is competitive but CAGRA support is limited
- Vulkan covers devices not covered by other backends

## Consequences

- SYCL backend compiles only when `ELIPS_GPU_SYCL=ON` and IntelSYCL is found
- SYCL kernels mirror CUDA/HIP kernels structurally
- Users with Intel Arc GPUs get 2x speedup (SYCL) vs CPU
- Cross-backend compilation via adaptiveCpp is documented but not the default path