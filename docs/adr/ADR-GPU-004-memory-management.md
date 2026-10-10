# ADR-GPU-004: Memory Management Strategy

**Status:** Accepted (amended 2026-10-11)
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

Discrete GPU device memory is scarce (8–80 GB). Per-request allocations cause
fragmentation and overhead. A pool allocator is needed for performance and
predictability on PCIe GPUs.

## Decision

Use a best-fit pool allocator (`GpuMemoryManager`) that pre-allocates a buffer
from device memory and sub-allocates from it. Pinned host memory is managed
separately for fast CPU↔GPU transfers.

Components:
1. `GpuMemoryManager` — pool allocator with best-fit free-block selection, thread-safe
2. `GpuMemoryPool` — simpler fixed-size pool for index storage
3. `PinnedBuffer` — RAII wrapper for pinned host memory (async DMA)
4. `UnifiedBuffer` — zero-copy shared memory on Apple Silicon / Jetson

## Reality Update (2026-10-11)

- Config defaults (`GpuConfig`): `device_memory_pool_bytes` defaults to **0**
  (derive at init, not a fixed 80%); `pinned_host_pool_bytes` defaults to
  **256 MB**.
- **On the validated Metal path the pool is largely moot**: every Metal buffer is
  `MTLResourceStorageModeShared`, so CPU and GPU share one allocation and there is
  no device↔host blit — "transfer" is a `memcpy` over `[buffer contents]`. The
  best-fit pool and pinned-host machinery matter mainly for the **experimental**
  discrete CUDA/HIP backends.
- Pool exhaustion maps to `GpuError::InsufficientMemory`, which (per ADR-GPU-008)
  surfaces as a **logged** demotion or a typed error — not a silent swallow.

## Consequences

- Pinned host pool: 256 MB default.
- Unified memory bypasses copies entirely on Apple Silicon (zero-copy).
- The pool allocator is validated by unit tests (`GpuMemoryManagerTest`); its
  production exercise on large discrete GPUs is **unvalidated** (CUDA/HIP experimental).
