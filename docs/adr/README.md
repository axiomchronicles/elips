# Architecture Decision Records

Each ADR captures one significant decision, its context, and the trade-offs accepted.

| ADR | Title | Status |
|---|---|---|
| [0001](ADR-0001-cpp23-core.md) | C++23 as the core language | Accepted |
| [0002](ADR-0002-embedded-only.md) | Embedded-only deployment (no server) | Accepted |
| [0003](ADR-0003-hnsw-primary-index.md) | HNSW as the primary ANN index | Accepted |
| [0004](ADR-0004-wal-format.md) | Record-based WAL with CRC32C | Accepted |
| [0005](ADR-0005-isolation-level.md) | Atomic batched transactions for v1.0 | Accepted |
| [0006](ADR-0006-pybind11.md) | PyBind11 for Python bindings | Accepted |
| [0007](ADR-0007-snapshot-plus-wal.md) | Snapshot + WAL persistence | Accepted |
| [0008](ADR-0008-file-locking.md) | File advisory locking for coordination | Accepted |
| [0009](ADR-0009-dynamic-schema.md) | Dynamic metadata schema | Accepted |
| [0010](ADR-0010-eql-language.md) | EQL as a parsed language | Accepted |

## GPU ADRs

The GPU design is Metal-first. Metal is hardware-validated on Apple M4; CUDA and
HIP are implemented but **experimental and not hardware-validated**. Several
early GPU ADRs over-claimed cuVS/CAGRA and a silent fallback; they are corrected
below. See the [GPU guide](../gpu/index.md) for the authoritative maturity table.

| ADR | Title | Status |
|---|---|---|
| [GPU-001](ADR-GPU-001-multi-backend-strategy.md) | Multi-backend abstraction (`GpuBackend`) | Accepted (amended; SYCL/Vulkan dropped) |
| [GPU-002](ADR-GPU-002-cuvs-vs-custom-kernels.md) | cuVS vs custom CUDA kernels | Superseded (cuVS never built; cuBLAS + custom only) |
| [GPU-003](ADR-GPU-003-gpu-build-cpu-serve.md) | GPU build / CPU serve (CAGRA export) | Superseded (CAGRA never built) |
| [GPU-004](ADR-GPU-004-memory-management.md) | GPU memory management | Accepted (amended) |
| [GPU-005](ADR-GPU-005-dynamic-batching.md) | Dynamic batching (`QueryBatcher`) | Accepted (amended, renamed) |
| [GPU-006](ADR-GPU-006-apple-unified-memory.md) | Unified memory on Apple Silicon | Accepted (validated on M4) |
| [GPU-007](ADR-GPU-007-precision-strategy.md) | Precision selection | Accepted (amended; FP16/Int8 compute not implemented) |
| [GPU-008](ADR-GPU-008-fallback-chain.md) | Fallback chain (logged, no-silent) | Accepted (amended) |
