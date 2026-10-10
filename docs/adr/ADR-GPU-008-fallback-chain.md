# ADR-GPU-008: Fallback Chain Design

**Status:** Accepted (amended 2026-10-11 — fallback is now **logged, never silent**)
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

GPU operations can fail: device not found, insufficient memory, kernel launch
error, device lost. Each failure needs a defined CPU fallback to maintain
availability — but a *silent* downgrade hides correctness/perf cliffs from the
user.

## Decision (as amended)

**No-silent-fallback.** Device selection is explicit, and any demotion to CPU is
logged.

- `device=` selects `cpu` / `metal` / `cuda:N` / `hip:N` / `auto`.
- A **named** backend (`metal`/`cuda`/`hip`, `GpuPolicy::Specific`) that is absent
  **raises `BackendUnavailableError` at open time** — unless `fallback=True`
  (`GpuConfig::allow_cpu_fallback`), which demotes to CPU and **logs a warning**:
  `[elips] warning: GPU backend '<name>' unavailable; falling back to CPU`.
- `auto` (`GpuPolicy::Auto`) always permits CPU fallback (CPU is always present),
  and still logs when it had to demote.
- At query time, a backend error (`InsufficientMemory`, `KernelLaunchFailed`, …)
  degrades to the CPU exact/HNSW path; the event is counted in
  `GpuMetricsSnapshot::fallback_events_total`.

The runtime chain, simplest-always-available last:

```
GPU index/bf_knn
  ↓ failure (logged, counted)
CPU IndexHNSW (approximate) or IndexFlat (exact, always available)
```

## Reality Update (2026-10-11)

- The old wording — "silently degrades", "transparent to user" — is **wrong** and
  has been corrected above. Fallback is observable: a stderr warning at open and a
  counter at query time.
- `GpuPolicy::RequireGpu` is effectively **not a distinct runtime guarantee**: it
  does not pin the GPU for every query. The meaningful control is `device=` +
  `fallback=`. Treat `RequireGpu` as legacy; prefer a named `device=` with
  `fallback=False` to hard-fail when the GPU is missing.

## Consequences

- Users get a clear error or a logged warning instead of a mystery slowdown.
- CPU `IndexFlat` provides exact ground truth when correctness matters.
- `fallback_events_total` lets callers detect silent-perf-cliff conditions in
  metrics.
