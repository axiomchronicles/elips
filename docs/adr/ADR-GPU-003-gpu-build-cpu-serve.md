# ADR-GPU-003: GPU Build / CPU Serve Mode

**Status:** Superseded (2026-10-11) — depended on CAGRA, which was never built
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

GPU index construction (CAGRA) was projected to be far faster than CPU HNSW
build, while online queries are cheap on CPU HNSW. A "build on GPU, serve on CPU"
mode was proposed to get fast builds without requiring a GPU at query time.

## Original Decision (not realized)

`GpuBuild_CpuServe` would build a CAGRA graph on GPU, export it to HNSW format,
and serve queries from the existing CPU index.

## Reality Update (2026-10-11) — this ADR is superseded

The `IndexBuildMode` enum (`GpuBuild_CpuServe`, `GpuBuild_GpuServe`, `Hybrid`)
still exists in `GpuConfig`, and the **default is `GpuBuild_GpuServe`**, not
`GpuBuild_CpuServe`. But the mode as described hinged on **CAGRA build + CAGRA→HNSW
export, which were never implemented** (see ADR-GPU-002). There is therefore no
validated "8–12x GPU build then CPU serve" path.

What is real today:

- **Metal (validated):** GPU distance + GPU top-k + GPU IVF scan run at *query*
  time; `bf_knn` does exact GPU k-NN with no index build. The validated win is
  on the serve side (batched search), not a CAGRA build export.
- **CPU:** the CPU `IndexHNSW` / IVF family builds and serves on CPU as always,
  independent of any GPU.

The CAGRA-export mode remains unimplemented scaffolding. If a GPU-build path is
ever shipped, it will get a fresh ADR.

## Consequences

- Do not rely on `GpuBuild_CpuServe` / CAGRA export — it is not implemented.
- `GpuHybridIndex` wraps a GPU index with a CPU fallback for the paths that are real.
- The honest GPU story is query-time acceleration (Metal); see the
  [GPU guide](../gpu/index.md) and [benchmarks](../overview/choosing-an-index.md).
