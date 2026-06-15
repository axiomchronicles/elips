# ADR-GPU-004: Memory Management Strategy

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

GPU device memory is scarce (8-80 GB). Per-request allocations cause fragmentation and overhead. A pool allocator is needed for performance and predictability.

## Decision

Use a best-fit pool allocator (`GpuMemoryManager`) that pre-allocates a single large buffer from device VRAM and sub-allocates from it. Pinned host memory is managed separately for fast CPU↔GPU transfers.

Components:
1. `GpuMemoryManager` — pool allocator with best-fit free block selection, thread-safe
2. `GpuMemoryPool` — simpler fixed-size pool for index storage
3. `PinnedBuffer` — RAII wrapper for pinned host memory (used for async DMA)
4. `UnifiedBuffer` — zero-copy shared memory on Apple Silicon / Jetson

## Rationale

- Pre-allocation avoids per-kernel `cudaMalloc` overhead
- Best-fit reduces fragmentation vs first-fit
- Thread-safe for concurrent search queries
- Pinned buffers enable async `cudaMemcpyAsync` transfers
- Unified memory bypasses copy entirely on Apple Silicon (zero-copy)

## Consequences

- Default pool size: 80% of free VRAM (configurable via `GpuConfig::device_memory_pool_bytes`)
- Pinned host pool: 256 MB default
- Pool exhaustion returns `GpuError::InsufficientMemory`, triggering CPU fallback