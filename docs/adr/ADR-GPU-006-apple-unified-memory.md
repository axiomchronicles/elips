# ADR-GPU-006: Unified Memory on Apple Silicon

**Status:** Accepted (validated on Apple M4, 2026-10-11)
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

Apple M-series chips use a unified memory architecture where CPU and GPU share
physical memory. PCIe GPUs require explicit `cudaMemcpy`/upload/download. On
Apple Silicon those copies are unnecessary — the GPU reads CPU-allocated buffers
directly.

## Decision

Use `MTLResourceStorageModeShared` for all Metal buffers on Apple Silicon.
Treat upload/download as a `memcpy` over `[buffer contents]`, eliminating copy
overhead and staging buffers.

Detection: `device.hasUnifiedMemory`. Auto-enabled on Apple Silicon.

## Reality Update (2026-10-11)

- This is **implemented and validated** on Apple M4. `MetalBackend` allocates
  every buffer `MTLResourceStorageModeShared` (`MetalBackend.mm`), so distance,
  top-k, and IVF-scan kernels read host-written data with no blit.
- `GpuConfig::use_unified_memory` defaults to **false** and is auto-promoted when
  the device reports `hasUnifiedMemory`; `GpuDeviceInfo::has_unified_memory`
  exposes it for observability.
- This zero-copy transfer is the single biggest simplification Metal buys over
  CUDA, and is why ELIPS's Metal cloner/transfer paths are pointer reads rather
  than DMA.

## Consequences

- `MetalBackend` upload/download use `std::memcpy` on unified memory.
- Larger-than-"VRAM" indexes are practical on Apple Silicon (one shared pool).
- Non-unified GPUs would fall back to standard blit copies (not exercised on the
  validated path).
