# ADR-GPU-008: Fallback Chain Design

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

GPU operations can fail for many reasons: device not found, insufficient memory, kernel launch error, device lost. Each failure must have a defined CPU fallback to maintain service availability.

## Decision

Multi-level fallback chain:

```
Level 0: GPU (CAGRA/IVF-PQ/BruteForce)
  ↓ failure
Level 1: GPU BruteForce (if supported; exact results, no index needed)
  ↓ failure
Level 2: CPU HierarchicalGraphIndex (HNSW, approximate)
  ↓ failure
Level 3: CPU ExactIndex (brute force, exact, always available)
```

Fallback triggers:
- `GpuError::DeviceNotFound` → Level 2
- `GpuError::InsufficientMemory` → Level 2 (log warning)
- `GpuError::KernelLaunchFailed` → Level 1 (retry brute-force) or Level 2
- `GpuError::IndexBuildFailed` → Level 2

## Rationale

- Always serve the query, never return an error to the user
- GPU Index fails → try simpler GPU path → fall to CPU
- CPU HNSW is always available (built into ELIPS core)
- CPU ExactIndex provides ground truth when correctness matters

## Consequences

- `GpuPolicy::Auto` silently degrades on any GPU failure
- `GpuPolicy::RequireGpu` fails `open()` if no GPU found, but still falls back at query time
- Fallback events are counted in `GpuMetricsSnapshot::fallback_events_total`
- Fallback is transparent to user — same API, same results, possibly slower