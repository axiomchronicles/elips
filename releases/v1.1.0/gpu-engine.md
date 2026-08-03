# GPU Acceleration & Suballocator — v1.1.0 Updates

This document detail the GPU build system gating (**F8**) and GPU memory pool suballocator corrections (**F9**) shipped in ELIPS v1.1.0.

---

## F8 — Cross-Platform GPU Build Gating

### Problem Definition
In pre-v1.1.0 release CMake configurations:
- `ELIPS_GPU_METAL` defaulted to `ON` across all target operating systems.
- Running standard CMake configuration (`cmake -S . -B build`) on a Linux or Windows host defined `ELIPS_GPU_ENABLED`, compiled all 17 source files of the `elips_gpu` static library, and exposed GPU symbols through headers and Python bindings despite no underlying Metal runtime existing on non-Apple hardware.
- Option declarations were duplicated across top-level `CMakeLists.txt` and `src/gpu_engine/CMakeLists.txt`, causing cache conflicts depending on invocation order.

### Implementation & Fix
1. **Host-Aware Default Gating:**
   `ELIPS_GPU_METAL` now defaults strictly to `${APPLE}` (`ON` on macOS, `OFF` on Linux/Windows).
2. **Explicit Warning & Fallback:**
   If a user explicitly forces `-DELIPS_GPU_METAL=ON` on a non-Apple host, CMake emits a warning and disables the backend unless valid Metal cross-toolchains are detected.
3. **Single-Source Option Definitions:**
   Removed duplicate `option()` definitions in `src/gpu_engine/CMakeLists.txt`. The top-level build manifest is now the single source of truth.
4. **CI Verification:**
   Added a dedicated CI job asserting that default Linux builds exclude the GPU static library and export `_has_gpu = False` in Python bindings.

```bash
# Linux Build with CUDA Backend
cmake -S . -B build -DELIPS_GPU_CUDA=ON

# Apple Build (Metal Enabled Automatically)
cmake -S . -B build
```

---

## F9 — GPU Pool Suballocator Leak & Fragmentation Fixes

### Problem Definition
`GpuMemoryManager` implements a best-fit suballocator over raw device memory pools (80% of VRAM by default).

Auditing identified three critical bugs in the suballocator logic:

1. **Un-tracked Remainder Leaks:**
   When allocating memory from a best-fit free block larger than the requested size, the suballocator handed the requested pointer to the caller, but discarded the remaining leftover span! The remainder was neither tracked as live nor returned to the free list, permanently leaking device memory for the application lifetime.

2. **Un-coalesced Free List Fragmentation:**
   `deallocate()` returned freed blocks to the free list without merging adjacent free spans. Repeated allocation and deallocation of varying tensor sizes fragmented VRAM into tiny disjoint blocks, causing `InsufficientMemory` errors despite gigabytes of total free pool space remaining.

3. **Inaccurate `bytes_available()` Reporting:**
   `bytes_available()` computed available space as `total_pool_bytes - allocated_bytes`. Because leaked remainders were not counted in `allocated_bytes`, `bytes_available()` reported high free memory when usable contiguous memory was actually zero.

### Implementation & Fix

1. **Immediate Remainder Splitting:**
   When allocating from a free block larger than requested (+ alignment padding), the remainder block is immediately sliced off and returned to the `free_blocks_` vector.

2. **Live Allocation Map (`live_blocks_`):**
   `GpuMemoryManager` now maintains a `std::unordered_map<void*, LiveBlock>` tracking exact rounded sizes and backend root allocation IDs (`root` index). This enables size lookup on `deallocate()` and guards against double-free corruption.

3. **Adjacent Span Coalescing (`release_locked`):**
   Deallocation passes the block to `release_locked()`, which scans `free_blocks_` for physically adjacent free blocks belonging to the **same backend root allocation** and merges them into a single contiguous free block.

4. **Accurate Available Memory Reporting:**
   `bytes_available()` now returns the exact sum of available free block spans plus uncommitted pool headroom.

```cpp
// Internal GpuMemoryManager Architecture (v1.1.0)
struct LiveBlock {
    size_t bytes; // Exact rounded size reserved
    size_t root;  // Originating backend allocation ID
};

struct FreeBlock {
    void* ptr;
    size_t bytes;
    size_t root;  // Only merge free blocks sharing the same root ID
};
```

---

## Python GPU API Surface (`elips._modern.gpu`)

v1.1.0 exposes a modern, pythonic GPU wrapper under `elips._modern.gpu`:

### Introspection & Discovery
- `elips.accelerators()` — Returns list of available GPU devices (`AcceleratorSpec`).
- `elips.accelerator(config=None)` — Selects optimal GPU accelerator matching config.

### High-Level `Accelerator` Interface
- `acc.reserve(pool_bytes=0)` — Sizes memory pool up front (0 = 80% VRAM).
- `acc.search(queries, corpus, top=5, metric="cosine")` — Batch vector search.
- `acc.distances(queries, corpus, metric="cosine")` — Returns `DistanceMatrix`.
- `acc.nearest(distances, top=5)` — Selects top-K indices and distances.
- `acc.memory_usage()` — Returns `(bytes_used, bytes_available, peak_bytes_used)`.

### Code Example: End-to-End GPU Acceleration in Python

```python
import elips

# Probe available GPUs
specs = elips.accelerators()
for spec in specs:
    print(
        f"Device: {spec.name}, Backend: {spec.backend}, VRAM: {spec.memory_gb:.2f} GB"
    )

# Initialize best GPU
gpu = elips.accelerator()
if gpu is None:
    print("No compatible GPU found; falling back to CPU SIMD.")
else:
    with gpu:
        # Pre-reserve 512 MB pool
        gpu.reserve(512 * 1024 * 1024)

        corpus = [[0.1, 0.2, 0.9, 0.4], [0.9, 0.1, 0.0, 0.2], [0.0, 0.8, 0.1, 0.5]]
        queries = [[0.12, 0.18, 0.85, 0.41], [0.88, 0.11, 0.02, 0.19]]

        # Execute GPU Batch Search
        top_indices, top_distances = gpu.search(queries, corpus, top=2, metric="cosine")

        for q_idx, (indices, dists) in enumerate(zip(top_indices, top_distances)):
            print(f"Query {q_idx} hits:", list(zip(indices, dists)))

        # Inspect accurate memory usage (F9 Fix)
        used, avail, peak = gpu.memory_usage()
        print(
            f"Pool VRAM: Used={used / 1e6:.1f}MB, Avail={avail / 1e6:.1f}MB, Peak={peak / 1e6:.1f}MB"
        )
```
