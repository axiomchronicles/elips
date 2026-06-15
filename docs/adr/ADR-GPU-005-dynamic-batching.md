# ADR-GPU-005: Dynamic Batching Strategy

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

Online vector search requests arrive one at a time from user code. Launching one GPU kernel per query is inefficient — kernel launch overhead dominates. Coalescing concurrent queries into a single batch kernel launch dramatically improves throughput.

## Decision

Use a **push model** `DynamicBatcher` with a configurable time window. Queries accumulate in a queue; when the window expires or the batch is full, all queries are dispatched in one kernel launch.

```
Thread 1: seek(q1) ─────┐
Thread 2: seek(q2) ──────┤ DynamicBatcher ──→ GPU kernel [q1, q2, q3] → results
Thread 3: seek(q3) ──────┘    (500μs window)
```

## Rationale

- Push model (queries pushed into batcher, results returned via `std::future`)
- Configurable window (default 500µs) balances latency vs throughput
- Max batch size (default 256) prevents unbounded latency
- Background worker thread flushes on window expiry or batch fullness
- Stats available for observability (batch sizes, coalescing ratio)

## Consequences

- Single-query latency increases by up to window_us (500µs typical)
- Throughput improves 3-10x for concurrent workloads
- CPU fallback path is unaffected — batcher is only active when GPU is available
- `GpuConfig::dynamic_batch_window_us` and `dynamic_batch_max_size` expose tuning