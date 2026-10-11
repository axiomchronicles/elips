# ADR-GPU-007: Precision Selection Strategy

**Status:** Accepted (amended 2026-10-11 — FP16/Int8 compute not yet implemented)
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

Modern GPUs support FP32, FP16, BF16, INT8, each trading accuracy for speed and
memory. Users should not need to understand GPU-specific precision capabilities.

## Decision

Auto-select precision from hardware capability and user preference, overridable
via `GpuConfig::search_precision`. Storage is always FP32 (lossless ground
truth).

Intended auto-selection:
- `GpuPrecision::Auto` (default): FP16 if available, else FP32
- `GpuPrecision::FP32`: always FP32 (baseline)
- `GpuPrecision::FP16`: FP16 if supported, else warn + FP32
- `GpuPrecision::Int8`: INT8 for ultra-fast approximate search

## Reality Update (2026-10-11)

- **The validated path runs all math in FP32.** On Metal, FP16 is *storage-only*
  for centroids and is accepted-then-ignored for compute; there is **no FP16 or
  INT8 compute kernel**. `GpuConfig::enable_fp16_search` defaults to **false**.
- `GpuPrecision::Auto` therefore effectively resolves to **FP32** today.
  `GpuDeviceInfo` still reports `supports_fp16` for observability, but selecting
  FP16/Int8 does not change the kernels that run.
- Mixed-precision "FP16 compute → FP32 rerank" is **unimplemented aspiration**,
  not a shipped pipeline.

## Consequences

- Treat `search_precision` other than FP32 as a no-op on the validated backend.
- FP16/Int8 compute kernels are future work (and would land first on Metal, where
  parity tests exist); CUDA/HIP precision paths are experimental and unvalidated.
