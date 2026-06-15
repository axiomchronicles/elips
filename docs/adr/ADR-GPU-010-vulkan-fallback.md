# ADR-GPU-010: Vulkan Compute as Universal Fallback

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

Some GPUs do not have CUDA, ROCm, Metal, or SYCL support: older NVIDIA/AMD cards, ARM Mali, Qualcomm Adreno, PowerVR, and any Vulkan 1.2+ capable device. A universal compute backend ensures ELIPS runs on the widest range of hardware.

## Decision

Use Vulkan Compute with pre-compiled SPIR-V shaders as the universal fallback backend. All GLSL compute shaders are compiled to SPIR-V at build time using `glslangValidator`. No runtime shader compilation.

## Tradeoffs

**Advantages:**
- Runs on virtually any modern GPU (Vulkan 1.2+)
- Pre-compiled SPIR-V avoids runtime shader compilation overhead
- GLSL compute shaders are simple and well-understood

**Disadvantages:**
- No advanced ANN index (CAGRA, IVF-PQ) — brute-force only
- No specialized BLAS (uses custom GEMM kernel, not cuBLAS/rocBLAS)
- ~1.5x speedup vs CPU (modest compared to 4-8x on CUDA)
- Requires Vulkan SDK at build time
- Shader compilation adds build complexity

## Consequences

- Vulkan backend is lowest priority in `GpuSelector` ranking
- Only `GpuIndexAlgorithm::BruteForce` is supported
- SPIR-V binaries are checked into the repository (deterministic builds)
- Users with Vulkan-only GPUs get ~1.5x improvement over CPU SIMD path