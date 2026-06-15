# ADR-GPU-007: Precision Selection Strategy

**Status:** Accepted
**Date:** 2026-05-31
**Deciders:** ELIPS Engineering

## Context

Modern GPUs support multiple floating-point precisions: FP32, FP16, BF16, INT8. Each offers different accuracy/speed/memory tradeoffs. Users should not need to understand GPU-specific precision capabilities.

## Decision

Auto-select precision based on hardware capabilities and user preference. User can override via `GpuConfig::search_precision`.

Auto-selection logic:
- If `GpuPrecision::Auto` (default): FP16 if Tensor Cores/ANE available, else FP32
- If `GpuPrecision::FP32`: always FP32 (maximum accuracy, baseline)
- If `GpuPrecision::FP16`: FP16 if hardware supports it, else warn and fallback to FP32
- If `GpuPrecision::Int8`: INT8 for ultra-fast approximate search (datacenter GPUs)

Storage always uses FP32 (lossless ground truth). Mixed-precision pipeline: FP32 storage → FP16 GPU upload → FP16 GPU compute → FP32 reranking → FP32 results.

## Rationale

- FP16 provides 2x memory savings and 2-4x throughput on Tensor Core GPUs
- FP32 is always available as correctness baseline
- Auto mode picks optimal precision without user intervention
- Reranking top-2K candidates in FP32 ensures result accuracy

## Consequences

- Each backend must implement FP16 kernels (FP16 native on Metal/Apple7+, CUDA Tensor Cores, HIP Matrix Cores)
- GpuDeviceInfo reports `supports_fp16`, `supports_bf16`, `supports_int8`
- Users see `fp16_search_enabled` in metrics