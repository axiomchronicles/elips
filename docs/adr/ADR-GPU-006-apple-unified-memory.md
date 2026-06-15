# ADR-GPU-006: Unified Memory on Apple Silicon

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

Apple M-series chips use a unified memory architecture where CPU and GPU share physical memory. Traditional PCIe-based GPUs require explicit `cudaMemcpy`/`upload`/`download` calls. On Apple Silicon, these copies are unnecessary — the GPU can read directly from CPU-allocated buffers.

## Decision

Use `MTLResourceStorageModeShared` for all Metal buffers on Apple Silicon. Provide a `UnifiedBuffer` abstraction that wraps a single allocation accessible from both CPU and GPU, eliminating copy overhead.

Detection: `device.hasUnifiedMemory` property on `MTLDevice`. Auto-enabled for all Apple Silicon Macs.

## Rationale

- Zero-copy eliminates PCIe transfer latency (typically 5-10 μs per transfer)
- Reduces memory footprint (one buffer instead of host + device copies)
- Simplifies code — no upload/download needed for unified buffers
- Graceful fallback: non-unified GPUs (AMD discrete on Mac Pro) use standard `MTLBlitCommandEncoder` copies

## Consequences

- `MetalBackend::upload()` and `download()` use `std::memcpy` on unified memory
- `GpuConfig::use_unified_memory` defaults to `true` on Apple Silicon
- `hasUnifiedMemory` flag in `GpuDeviceInfo` for observability