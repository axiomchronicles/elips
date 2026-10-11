# Phase 1C — GPU Engine Real Status

Scope audited: `include/elips/gpu_engine/*.hpp`, `src/gpu_engine/**`, `tests/{unit,integration}/gpu/*`, `tests/parity/*`, `benchmarks/gpu/*`, `bindings/python/elips/gpu/*`. Evidence is cited `file:line`. No claim that any backend "works" — only what the source statically shows.

## Headline findings (read first)

- **Metal is the ONLY backend with a real implementation.** `MetalBackend.mm` (272 LOC) is genuine Objective-C++ Metal code and compiles shaders at runtime from an inline string (`MetalBackend.mm:64-124`).
- **CUDA, HIP, SYCL, Vulkan are pure stubs.** Every one of their backend `.cpp` files is nothing but a `#error` guarded by the enable macro (`CudaBackend.cpp:2-4`, `HipBackend.cpp:2-4`, `SyclBackend.cpp:2-4`, `VulkanBackend.cpp:2-4`). Enabling the backend makes the file refuse to compile by design.
- **Those four backends cannot build even if the SDK is present.** CUDA/HIP CMake reference source files that do not exist (`CudaBackend.cu`, `HipBackend.hip`); SYCL/Vulkan reference files that are `#error` stubs. See build table.
- **The real `.metal` shader file is dead code.** `shaders/elips_distance_kernels.metal` (111 LOC, 5 real kernels) is never named in any `CMakeLists.txt` and never loaded by `MetalBackend.mm` — the backend uses a *different*, inline 3-kernel string instead.
- **Vulkan/CUDA kernel files are real-looking GLSL/CUDA but unreachable** (their backends are `#error` stubs). The Vulkan cosine shader also has a shadowing bug (`float dv = d[...]` where `d` is both the buffer and the loop var — `cosine_fp32.comp:24,30`), so it would not even compile.
- **`GpuKernelPort` and `GpuStreamPort` have ZERO implementers** anywhere in the tree. They are orphan interfaces.
- **No unit test runs real GPU code.** Unit tests inject `RecordingBackend` (a CPU emulator). Only the integration/parity tests touch a real backend, and they `GTEST_SKIP()` when no device is present — i.e. they only ever exercise Metal, only on Apple hardware with `ELIPS_GPU_METAL` on.

## Port method matrix

The five "port" headers. Only `GpuPort` is implemented by backends; the others are implemented (or not) elsewhere.

### `GpuPort` (`GpuPort.hpp:40-86`) — the backend interface
| Method | Signature site |
|---|---|
| `initialize(GpuConfig)` | :44 |
| `shutdown()` | :47 |
| `device_info()` | :49 |
| `is_available()` | :50 |
| `allocate_device(bytes)` | :52 |
| `free_device(GpuBuffer)` | :57 |
| `upload(host_src, dst, bytes)` | :59 |
| `download(src, host_dst, bytes)` | :62 |
| `compute_distances_batch(queries, database, out, nq, nb, dim, metric)` | :65 |
| `top_k(distances, idx_out, val_out, nq, nb, k)` | :75 |
| `synchronize()` | :84 |
| `is_idle()` | :85 |

### `GpuKernelPort` (`GpuKernelPort.hpp:12-36`)
| Method | Site | Implementers |
|---|---|---|
| `cosine_fp32(...)` | :16 | **NONE** |
| `euclidean_fp32(...)` | :23 | **NONE** |
| `dot_product_fp32(...)` | :30 | **NONE** |

Orphan interface — no class derives from `GpuKernelPort`.

### `GpuMemoryPort` (`GpuMemoryPort.hpp:12-29`)
| Method | Site | Implementer |
|---|---|---|
| `allocate(bytes, alignment=256)` | :16 | `GpuMemoryManager` |
| `deallocate(GpuBuffer&&)` | :19 | `GpuMemoryManager` |
| `allocate_pinned(bytes)` | :21 | `GpuMemoryManager` |
| `deallocate_pinned(ptr)` | :24 | `GpuMemoryManager` |
| `bytes_used()` | :26 | `GpuMemoryManager` |
| `bytes_available()` | :27 | `GpuMemoryManager` |
| `peak_bytes_used()` | :28 | `GpuMemoryManager` |

Implemented host-side by `GpuMemoryManager` (not by any device backend).

### `GpuStreamPort` (`GpuStreamPort.hpp:9-16`)
| Method | Site | Implementers |
|---|---|---|
| `synchronize()` | :13 | **NONE** |
| `is_complete()` | :14 | **NONE** |
| `wait_for_completion()` | :15 | **NONE** |

Orphan interface — no implementers.

### `GpuIndexPort` (`GpuIndexPort.hpp:20-45`) — extends `IndexPort` + `IndexTransferPort`
| Method | Site |
|---|---|
| `build_from_batch(vectors, ids, params)` | :24 |
| `search_batch(queries, k, ef_search)` | :30 |
| `export_to_cpu_index(cpu_index_out)` | :36 |
| `import_from_cpu_index(cpu_index)` | :39 |
| `device_bytes_used()` | :42 |
| `backend_name()` | :44 |

Implementers: `GpuBruteForceIndex`, `GpuIVFFlatIndex`, `GpuIVFPQIndex`, `GpuGraphIndex`, `GpuHybridIndex`, `GpuDistributedIndex`. These are host-side index orchestrators that delegate math to an injected `GpuPort` (e.g. `GpuBruteForceIndex.cpp:63,69,109,116` call `allocate_device`/`upload`/`compute_distances_batch`/`top_k`). They are real logic but are only as "GPU" as the backend handed to them; with `RecordingBackend` they run entirely on CPU.

## Per-backend GpuPort method status

Legend: **Real** = genuine device code; **Stub(#error)** = file is a compile-time `#error`, class defined in header but never compiled; **CPU** = runs on host.

| GpuPort method | Metal | CUDA | HIP | SYCL | Vulkan |
|---|---|---|---|---|---|
| initialize | Real (`.mm:32`) | Stub(#error) | Stub(#error) | Stub(#error) | Stub(#error) |
| shutdown | Real (`.mm:130`) | Stub | Stub | Stub | Stub |
| device_info | Real (`.mm:137`) | Stub | Stub | Stub | Stub |
| is_available | Real (`.mm:138`) | Stub | Stub | Stub | Stub |
| allocate_device | Real (`.mm:140`) | Stub | Stub | Stub | Stub |
| free_device | Real (`.mm:151`) | Stub | Stub | Stub | Stub |
| upload | Real (`.mm:157`) | Stub | Stub | Stub | Stub |
| download | Real (`.mm:175`) | Stub | Stub | Stub | Stub |
| compute_distances_batch | Real GPU (`.mm:194`) | Stub | Stub | Stub | Stub |
| top_k | **CPU** (`std::partial_sort`, `.mm:248-265`) | Stub | Stub | Stub | Stub |
| synchronize | No-op (`.mm:267`) | Stub | Stub | Stub | Stub |
| is_idle | Returns `true` (`.mm:268`) | Stub | Stub | Stub | Stub |

Notes:
- **Metal `top_k` is NOT on the GPU** — it is a host `std::partial_sort` loop (`.mm:251-263`). The `top_k_reduce` kernel exists only in the dead `.metal` file (`elips_distance_kernels.metal:58-82`).
- **Metal `synchronize`/`is_idle` are no-ops** — distance kernels already block with `waitUntilCompleted` (`.mm:241`), so there is no async state to sync.
- CUDA header declares an extra `gemm_fp32` (`CudaBackend.hpp:41`) with no definition anywhere.
- CUDA/HIP/SYCL/Vulkan rows are "Stub" wholesale: the entire translation unit is `#ifdef <MACRO> #error #endif`, so none of the 12 methods has a body. The classes are declared in the `.hpp` but never defined.

## Kernel-source reality

| File | LOC | Status |
|---|---|---|
| `metal/MetalBackend.mm` inline shaders (`.mm:64-124`) | ~50 kernel LOC (3 kernels: cosine/euclidean/dot) | **Real & actually used** at runtime |
| `metal/shaders/elips_distance_kernels.metal` | 111 (5 kernels incl. top_k_reduce, pq_encode) | Real GLSL-ish Metal, but **DEAD** — not in any CMake, not loaded by `.mm` |
| `cuda/kernels/cosine_fp32.cu` | 30 | **Placeholder** — `#error` + commented-out "expected implementation" (`cosine_fp32.cu:6-29`) |
| `hip/kernels/cosine_fp32.hip` | 5 | **Placeholder** — bare `#error` (`cosine_fp32.hip:5`) |
| `sycl/kernels/cosine_fp32.cpp` | 4 | **Placeholder** — bare `#error` (`cosine_fp32.cpp:4`) |
| `vulkan/shaders/cosine_fp32.comp` | 54 | Real-looking GLSL w/ shared-mem reduction, but **has a compile bug** (buffer `d[]` shadowed by loop var `d`, `:24,30`) and is never compiled |
| `vulkan/shaders/euclidean_fp32.comp` | 41 | Real-looking GLSL, never compiled (no glslang build step wired) |
| `vulkan/shaders/dot_product_fp32.comp` | 40 | Real-looking GLSL, never compiled |

Total kernel/backend LOC counted: Metal 272 (.mm) + 111 (.metal, dead); Vulkan shaders 135; CUDA .cu 30; HIP .hip 5; SYCL .cpp 4.

## Build-flag summary

Root `CMakeLists.txt`:
- `ELIPS_GPU_METAL` default `${APPLE}` (:46); `ELIPS_GPU_CUDA/HIP/SYCL/VULKAN` default `OFF` (:47-50).
- Composite `ELIPS_GPU_ENABLED` is set ON if any backend is on (:58-60); when ON, builds `src/gpu_engine`, links `elips_gpu` into `elips_core`, and defines `ELIPS_GPU_ENABLED` publicly (:101-103). Objective-C++ is enabled on Apple (:32).
- Guard at :52-55 warns if `ELIPS_GPU_METAL` is set on non-Apple and compiles no Metal backend.

`src/gpu_engine/CMakeLists.txt`:
- `elips_gpu` STATIC lib built from 17 host-side `.cpp` files (:5-23) — none of the backend `.cpp` files are in this list.
- Each backend added via `add_subdirectory` + compile-definition only when its flag is on (:44-76). Metal links PUBLIC, the rest PRIVATE.
- `ELIPS_GPU_TESTS` option (default OFF) does `add_subdirectory(tests)` (:79-82) — **but `src/gpu_engine/tests/` does not exist**, so turning it on breaks configure.

Per-backend CMake — static compilability verdict:

| Backend | SDK gate | Sources listed | Compiles if SDK present? |
|---|---|---|---|
| Metal (`metal/CMakeLists.txt`) | `find_library Metal/MPS/Foundation REQUIRED` | `MetalBackend.mm` (exists) | **Yes** on Apple w/ Xcode. Flags `-fobjc-arc -O3 -ffast-math` (:18-19, dup'd :27-29) |
| CUDA (`cuda/CMakeLists.txt`) | `find_package(CUDAToolkit 12.0)`, else `return()` | `CudaBackend.cu`, `kernels/cosine_fp32.cu` | **No** — `CudaBackend.cu` does not exist (file is `CudaBackend.cpp`); and `.cu` is a `#error` anyway |
| HIP (`hip/CMakeLists.txt`) | `find_package(hip)`, else `return()` | `HipBackend.hip`, `kernels/cosine_fp32.hip` | **No** — `HipBackend.hip` does not exist (file is `HipBackend.cpp`); `--ffast-math` flag typo (:17) |
| SYCL (`sycl/CMakeLists.txt`) | `find_package(IntelSYCL)`, else `return()` | `SyclBackend.cpp`, `kernels/cosine_fp32.cpp` | **No** — both files are `#error` stubs; `-fsycl` set |
| Vulkan (`vulkan/CMakeLists.txt`) | `find_package(Vulkan 1.2)` + `glslangValidator`, else `return()` | `VulkanBackend.cpp` only | **No** — `VulkanBackend.cpp` is a `#error` stub; `.comp` shaders are never added to any build step despite the glslang gate |

So on a non-Apple machine with all flags off (the default), none of this device code compiles; `elips_gpu` is host-only orchestration + `GpuMemoryManager`. On Apple, Metal is the only backend that both compiles and runs.

## Test reality summary

| Test | Backend used | Runs real GPU? |
|---|---|---|
| `tests/unit/gpu/GpuBruteForceIndexTest.cpp` | `RecordingBackend` (`:5,15,40`) | No — CPU emulator |
| `tests/unit/gpu/GpuAdvancedIndexTest.cpp` | `RecordingBackend` | No |
| `tests/unit/gpu/GpuMemoryManagerTest.cpp` | `RecordingBackend` | No |
| `tests/unit/gpu/GpuDeviceManagerTest.cpp` | `RecordingBackend` | No |
| `tests/unit/gpu/m4_memory_manager_test.cpp` | `RecordingBackend` | No |
| `tests/unit/gpu/DynamicBatcherTest.cpp` | (host logic) | No |
| `tests/integration/gpu/MetalBackendTest.cpp` | real via `probe_all_devices()` | Only if a device is found; else `GTEST_SKIP` (`:35,50,61,...`). In practice = Metal on Apple only |
| `tests/integration/gpu/DatabaseGpuLifecycleProbe.cpp` | real via `probe_all_devices()` | Skips when probe empty (`:11`) |
| `tests/parity/CpuGpuRecallParityTest.cpp` | real via `probe_all_devices()` | `GTEST_SKIP "No GPU available"` when none (`:34,48`) |
| `benchmarks/gpu/BenchGpuSearch.cpp` | real via `probe_all_devices()` (`:51`) | Only if device present |

`RecordingBackend` (`tests/unit/gpu/TestRecordingBackend.hpp:16-147`) is a full `GpuPort` that stores uploads in a `std::vector<float>` and computes distances/top-k on the CPU via `elips::distance` and `std::partial_sort` (`:85-120`). All unit coverage of the GPU index/memory layer is really CPU coverage of host orchestration. No test anywhere forces a real kernel to run; the real-backend tests are opt-in-by-hardware and skip silently.

## Python layer

`bindings/python/elips/gpu/accelerator.py` is a typed facade over native `gpu_devices`/`gpu_select`/`gpu_can_fit_index`. It is import-safe with no GPU: `accelerators()` returns `[]` and `accelerator()` returns `None` on CPU-only/CPU-only builds (`:38-39,417-419,448-449`). It adds no backend logic — it only reflects whatever the native layer (hence Metal-or-nothing in practice) exposes.
