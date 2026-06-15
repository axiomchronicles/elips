# ADR-GPU-002: cuVS vs Custom CUDA Kernels

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

For the CUDA backend, we must decide whether to depend on NVIDIA cuVS (which provides CAGRA, IVF-PQ, etc.) or write our own CUDA kernels for distance computation.

## Decision

Both. cuVS is the primary path when available; custom hand-written CUDA kernels serve as fallback when cuVS is not installed.

Priority:
1. If cuVS >= 25.02 is found at build time, the CUDA backend uses cuVS for CAGRA index construction and IVF-PQ compression
2. If cuVS is not found, the CUDA backend falls back to hand-written kernels (cosine/euclidean/dot product, top-K reduction) compiled with nvcc
3. cuBLAS GEMM is used for brute-force exact search in both paths

## Rationale

- cuVS provides production-grade CAGRA with 8-12x build speedup over CPU HNSW
- Not all deployments will have cuVS (it's a separate installation)
- Custom kernels ensure ELIPS works on any CUDA-capable system
- Kernel code is small (~50 lines per metric) and manageable

## Consequences

- CUDA backend CMakeLists.txt checks for cuVS with `find_package(cuvs QUIET)`
- Custom kernels compile with `CMAKE_CUDA_ARCHITECTURES "70;80;86;89;90;100"`
- Both paths are gated behind `ELIPS_CUDA_ENABLED`