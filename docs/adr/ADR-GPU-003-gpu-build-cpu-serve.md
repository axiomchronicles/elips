# ADR-GPU-003: GPU Build / CPU Serve Mode

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

GPU index construction (CAGRA) is 8-12x faster than CPU HNSW build. However, online search queries are I/O-bound and single-threaded, where CPU HNSW already delivers sub-200µs latency. Requiring a GPU for every query adds cost and complexity.

## Decision

The `GpuBuild_CpuServe` mode builds the index on GPU, exports it to HNSW format, then serves all queries via the existing CPU `HierarchicalGraphIndex`.

Flow:
1. Batch of vectors → GPU CAGRA build → fast index construction
2. CAGRA graph → export to HNSW format (node vectors + neighbor edges)
3. Store as `index.hnsw` in segment directory (existing format)
4. CPU `HierarchicalGraphIndex` reads it normally for queries

## Rationale

- Index build: 8-12x faster (GPU)
- Search: unaffected (existing CPU HNSW path)
- GPU not required at query time — lower infrastructure cost
- Compatible with existing storage format — zero migration

## Consequences

- `GpuIndexPort::export_to_cpu_index()` must convert CAGRA graph → HNSW edges
- `GpuHybridIndex` wraps both GPU and CPU indexes, routing batch ops to GPU
- User selects mode via `GpuConfig::index_build_mode`