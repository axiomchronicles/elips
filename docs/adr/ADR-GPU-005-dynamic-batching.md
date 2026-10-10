# ADR-GPU-005: Dynamic Batching Strategy

**Status:** Accepted (amended 2026-10-11 — `DynamicBatcher` renamed `QueryBatcher`)
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

Online vector search requests arrive one at a time from user code. Launching one
GPU kernel per query is inefficient — kernel-launch overhead dominates.
Coalescing concurrent queries into a single batch kernel improves throughput.

## Decision

Use a **push model** `QueryBatcher` (formerly `DynamicBatcher`) with a
configurable time window. Queries accumulate in a queue; when the window expires
or the batch is full, all queries are dispatched in one kernel launch.

```
Thread 1: search(q1) ────┐
Thread 2: search(q2) ─────┤ QueryBatcher ──→ GPU kernel [q1, q2, q3] → results
Thread 3: search(q3) ─────┘    (500μs window)
```

## Rationale

- Push model (queries pushed into the batcher, results returned via `std::future`)
- Configurable window (default 500µs via `GpuConfig::dynamic_batch_window_us`)
- Max batch size (default 256 via `GpuConfig::dynamic_batch_max_size`)
- Background worker thread flushes on window expiry or batch fullness
- Stats available for observability (batch sizes, coalescing ratio)

## Reality Update (2026-10-11)

- The class is **`QueryBatcher`**; the `GpuConfig` field names
  (`dynamic_batch_window_us`, `dynamic_batch_max_size`) are unchanged.
- Batching is where the validated **Metal** win shows up most clearly: a single
  batched `bf_knn` call amortizes launch + sync and reaches ~914 QPS vs ~24 QPS
  for one-query-at-a-time on Apple M4 (see
  [benchmarks](../overview/choosing-an-index.md)).

## Consequences

- Single-query latency increases by up to `window_us` (500µs typical).
- Throughput improves substantially for concurrent / batched workloads.
- The CPU path is unaffected — the batcher is only active when a GPU is selected.
