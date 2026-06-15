# ELIPS Build System

This document describes the CMake build architecture for the ELIPS project, including all targets, dependencies, and build configuration options.

## Build System Overview

ELIPS uses **CMake 3.24+** with C++23 as the required language standard. The build system is structured as a hierarchical CMake project with a root `CMakeLists.txt` and subdirectories for the GPU engine and its backends.

## Dependency Graph

```
elips_cli ────────────┐
elips_bench ──────────┤
elips_gpu_bench ──────┤
elips_tests ──────────┤
elips_pymodule ───────┤
                      ▼
                 elips_core ──────────────┐
                      │                   │
                      │ (if GPU enabled)  │
                      ▼                   │
                  elips_gpu ──────────────┤
                      │                   │
          ┌───────────┼───────────┬───────┤
          ▼           ▼           ▼       ▼
   elips_gpu_metal  elips_gpu_cuda  ...  │
          │           │                   │
   Metal.framework  CUDA toolkit          │
   MPS.framework                         │
   Foundation.framework                   │
                                          │
   External dependencies:                 │
     GoogleTest (v1.15.2) ◄── tests      │
     PyBind11 (v2.13.6)   ◄── python      │
```

## Root CMakeLists.txt

The root `CMakeLists.txt` (`/CMakeLists.txt`) is the entry point:

```cmake
cmake_minimum_required(VERSION 3.24)
project(elips LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(APPLE)
    enable_language(OBJCXX)
endif()
```

### Key Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| C++ Standard | C++23 | Required for `std::expected`, `std::span`, three-way comparison |
| Extensions | OFF | Strict standard compliance |
| Default Build Type | `RelWithDebInfo` | Balanced optimization with debug symbols |
| OBJCXX (Apple) | Enabled | Required for Metal backend (`.mm` files) |

## Build Options

All build options are declared at the root level:

| Option | Default | Description |
|--------|---------|-------------|
| `ELIPS_GPU_METAL` | `ON` | Apple Metal GPU backend |
| `ELIPS_GPU_CUDA` | `OFF` | NVIDIA CUDA GPU backend |
| `ELIPS_GPU_HIP` | `OFF` | AMD ROCm/HIP GPU backend |
| `ELIPS_GPU_SYCL` | `OFF` | Intel oneAPI/SYCL GPU backend |
| `ELIPS_GPU_VULKAN` | `OFF` | Vulkan compute GPU backend |
| `ELIPS_BUILD_CLI` | `ON` | Build `elips` CLI tool |
| `ELIPS_BUILD_BENCH` | `ON` | Build benchmark suite |
| `ELIPS_BUILD_TESTS` | `ON` | Build test suite (GoogleTest) |
| `ELIPS_BUILD_PYTHON` | `OFF` | Build Python bindings (PyBind11) |

### GPU Enablement Logic

```cmake
set(ELIPS_GPU_ENABLED OFF)
if(ELIPS_GPU_CUDA OR ELIPS_GPU_HIP OR ELIPS_GPU_METAL OR ELIPS_GPU_SYCL OR ELIPS_GPU_VULKAN)
    set(ELIPS_GPU_ENABLED ON)
endif()

if(ELIPS_GPU_ENABLED)
    add_subdirectory(src/gpu_engine)
endif()
```

`ELIPS_GPU_ENABLED` is automatically set to `ON` if any GPU backend is enabled. This controls whether the GPU engine subdirectory is built and whether `elips_core` links against `elips_gpu`.

## Target: elips_core (Core Library)

```cmake
add_library(elips_core
    src/RecordID.cpp
    src/Metrics.cpp
    src/ExactIndex.cpp
    src/HierarchicalGraphIndex.cpp
    src/IndexFactory.cpp
    src/LockManager.cpp
    src/WAL.cpp
    src/Filter.cpp
    src/EQLLexer.cpp
    src/EQLParser.cpp
    src/QueryExecutor.cpp
    src/Database.cpp
)
target_include_directories(elips_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(elips_core PRIVATE -Wall -Wextra -Wpedantic)

if(ELIPS_GPU_ENABLED)
    target_link_libraries(elips_core PUBLIC elips_gpu)
    target_compile_definitions(elips_core PUBLIC ELIPS_GPU_ENABLED)
endif()
```

- **Type**: Static library (default `add_library` without `SHARED`).
- **Sources**: 12 `.cpp` files covering all core subsystems.
- **Public headers**: `include/` directory exposed to all dependents.
- **Compile options**: `-Wall -Wextra -Wpedantic` for strict warnings.
- **GPU linkage**: When GPU is enabled, `elips_core` publicly links `elips_gpu`, making GPU types available through the core headers. The `ELIPS_GPU_ENABLED` preprocessor definition enables GPU-specific code paths in headers like `Config.hpp` and `elips.hpp`.

### Source File Organization

| File | Subsystem | Purpose |
|------|-----------|---------|
| `RecordID.cpp` | Kernel | UUIDv7 generation, string conversion |
| `Metrics.cpp` | Vector Engine | Distance functions (scalar + NEON SIMD) |
| `ExactIndex.cpp` | Index Engine | Brute-force exact search |
| `HierarchicalGraphIndex.cpp` | Index Engine | HNSW ANN index |
| `IndexFactory.cpp` | Index Engine | Pluggable index creation |
| `LockManager.cpp` | Kernel | Advisory file locking |
| `WAL.cpp` | Storage | Write-ahead log (append, replay, CRC) |
| `Filter.cpp` | Metadata | Metadata predicate tree |
| `EQLLexer.cpp` | Query Engine | EQL tokenizer |
| `EQLParser.cpp` | Query Engine | EQL recursive-descent parser |
| `QueryExecutor.cpp` | Query Engine | EQL statement executor |
| `Database.cpp` | Core | ElipsInstance, Vault, Transaction, open(), checkpoint |

## Target: elips_cli (CLI Tool)

```cmake
option(ELIPS_BUILD_CLI "Build the elips command-line tool" ON)
if(ELIPS_BUILD_CLI)
    add_executable(elips_cli cli/src/main.cpp)
    set_target_properties(elips_cli PROPERTIES OUTPUT_NAME elips)
    target_link_libraries(elips_cli PRIVATE elips_core)
    target_compile_options(elips_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()
```

- **Output name**: `elips` (set via `OUTPUT_NAME` property).
- **Source**: Single `cli/src/main.cpp` (261 lines) with command parsing and JSON import/export.
- **Commands**: `info`, `vaults`, `query`, `checkpoint`, `export`, `import`, `verify`, `stats`, `bench`.

## Targets: elips_bench and elips_gpu_bench (Benchmarks)

```cmake
option(ELIPS_BUILD_BENCH "Build the benchmark suite" ON)
if(ELIPS_BUILD_BENCH)
    add_executable(elips_bench benchmarks/BenchMain.cpp)
    target_link_libraries(elips_bench PRIVATE elips_core)
    target_compile_options(elips_bench PRIVATE -Wall -Wextra -Wpedantic)

    if(ELIPS_GPU_ENABLED)
        add_executable(elips_gpu_bench benchmarks/gpu/BenchGpuSearch.cpp)
        target_link_libraries(elips_gpu_bench PRIVATE elips_core)
        target_compile_definitions(elips_gpu_bench PRIVATE ELIPS_GPU_ENABLED)
        target_compile_options(elips_gpu_bench PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()
```

- **`elips_bench`**: CPU benchmark suite. Measures insert throughput, search latency percentiles, recall@10 vs ExactIndex, and crash-recovery time. Self-contained.
- **`elips_gpu_bench`**: GPU search benchmark. Only built when GPU is enabled. Links against `elips_core` (which transitively brings `elips_gpu`).

## Target: elips_tests (Test Suite)

```cmake
option(ELIPS_BUILD_TESTS "Build the ELIPS test suite" ON)

if(ELIPS_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(elips_tests
        tests/unit/metrics_test.cpp
        tests/unit/exact_index_test.cpp
        tests/unit/record_id_test.cpp
        tests/unit/hnsw_recall_test.cpp
        tests/unit/eql_test.cpp
        tests/unit/vector_test.cpp
        tests/unit/config_test.cpp
        tests/unit/exact_edge_cases_test.cpp
        tests/unit/hnsw_edge_cases_test.cpp
        tests/unit/filter_edge_test.cpp
        tests/unit/scan_test.cpp
        tests/unit/durability_test.cpp
        tests/unit/eql_edge_test.cpp
        tests/integration/roundtrip_test.cpp
        tests/integration/transaction_filter_test.cpp
        tests/recovery/wal_recovery_test.cpp
        tests/concurrency/single_writer_test.cpp
        tests/concurrency/multi_reader_test.cpp
    )
    target_link_libraries(elips_tests PRIVATE elips_core GTest::gtest_main)

    if(ELIPS_GPU_ENABLED)
        target_sources(elips_tests PRIVATE
            tests/unit/gpu/GpuDeviceManagerTest.cpp
            tests/unit/gpu/GpuMemoryManagerTest.cpp
            tests/unit/gpu/DynamicBatcherTest.cpp
            tests/integration/gpu/MetalBackendTest.cpp
            tests/parity/CpuGpuRecallParityTest.cpp
        )
        target_compile_definitions(elips_tests PRIVATE ELIPS_GPU_ENABLED)
        if(APPLE AND ELIPS_GPU_METAL)
            target_link_libraries(elips_tests PRIVATE elips_gpu_metal)
            find_library(METAL_FW Metal)
            find_library(MPS_FW MetalPerformanceShaders)
            find_library(FOUNDATION_FW Foundation)
            target_link_libraries(elips_tests PRIVATE ${METAL_FW} ${MPS_FW} ${FOUNDATION_FW})
        endif()
    endif()

    include(GoogleTest)
    gtest_discover_tests(elips_tests)
endif()
```

### Test Organization

| Category | Tests | Purpose |
|----------|-------|---------|
| **Unit** | 13 files | Isolated component tests: metrics, indices, EQL, filters, vectors, config, durability |
| **Integration** | 2 files | Roundtrip (open→write→read→close→reopen→verify), transaction filters |
| **Recovery** | 1 file | WAL crash recovery scenario |
| **Concurrency** | 2 files | Single-writer test, multi-reader test |
| **GPU Unit** | 3 files | GpuDeviceManager, GpuMemoryManager, DynamicBatcher |
| **GPU Integration** | 1 file | MetalBackend end-to-end |
| **Parity** | 1 file | CPU vs GPU recall comparison |

### GoogleTest Integration

- **Version**: v1.15.2, fetched via `FetchContent` from GitHub.
- **Discovery**: `gtest_discover_tests()` registers each test case with CTest.
- **Static CRT**: `gtest_force_shared_crt=ON` for Windows compatibility.

## Target: elips_pymodule (Python Bindings)

```cmake
option(ELIPS_BUILD_PYTHON "Build the Python (PyBind11) bindings" OFF)

if(ELIPS_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(elips_pymodule bindings/python/elips_python.cpp)
    target_link_libraries(elips_pymodule PRIVATE elips_core)
    if(ELIPS_GPU_ENABLED)
        target_compile_definitions(elips_pymodule PRIVATE ELIPS_GPU_ENABLED)
    endif()
    set_target_properties(elips_pymodule PROPERTIES
        OUTPUT_NAME _core
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/elips
    )
endif()
```

- **PyBind11 version**: v2.13.6, fetched via `FetchContent`.
- **`pybind11_add_module()`**: Convenience function that sets up Python module compilation flags and linking.
- **Output**: `_core.cpython-*.so` in `bindings/python/elips/`.
- **GPU**: If enabled, `ELIPS_GPU_ENABLED` is defined, exposing GPU config types and device info.

## GPU Engine Build (src/gpu_engine/CMakeLists.txt)

```cmake
add_library(elips_gpu STATIC
    GpuDeviceManager.cpp
    GpuSelector.cpp
    GpuMemoryManager.cpp
    GpuMemoryPool.cpp
    DynamicBatcher.cpp
    GpuIngestionPipeline.cpp
    GpuSearchPipeline.cpp
    GpuQuantizationPipeline.cpp
    GpuProfiler.cpp
    GpuGraphIndex.cpp
    GpuIVFFlatIndex.cpp
    GpuIVFPQIndex.cpp
    GpuBruteForceIndex.cpp
    GpuHybridIndex.cpp
)

target_include_directories(elips_gpu
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal
            ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda
            ${CMAKE_CURRENT_SOURCE_DIR}/backends/hip
            ${CMAKE_CURRENT_SOURCE_DIR}/backends/sycl
            ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan
)

target_link_libraries(elips_gpu PUBLIC elips_core)
target_compile_options(elips_gpu PRIVATE -Wall -Wextra -Wpedantic)
```

- **Static library**: `elips_gpu` is always static.
- **14 source files**: Core GPU infrastructure + index implementations + pipelines.
- **Backend headers**: Private include paths for each backend's implementation headers (e.g., `MetalBackend.hpp`).
- **Links against `elips_core`**: Uses core types (IndexPort, Metric, Vector, RecordID).

### Per-Backend Subdirectories

Each GPU backend is in its own subdirectory:

```
src/gpu_engine/backends/
  metal/   → elips_gpu_metal   (Metal.framework, MPS.framework, Foundation.framework)
  cuda/    → elips_gpu_cuda    (CUDA toolkit)
  hip/     → elips_gpu_hip     (ROCm/HIP runtime)
  sycl/    → elips_gpu_sycl    (Intel oneAPI/SYCL)
  vulkan/  → elips_gpu_vulkan  (Vulkan SDK)
```

Backends are conditionally added:

```cmake
if(APPLE AND ELIPS_GPU_METAL)
    add_subdirectory(backends/metal)
    target_link_libraries(elips_gpu PUBLIC elips_gpu_metal)
    target_compile_definitions(elips_gpu PUBLIC ELIPS_METAL_ENABLED)
endif()

if(ELIPS_GPU_CUDA)
    add_subdirectory(backends/cuda)
    target_link_libraries(elips_gpu PRIVATE elips_gpu_cuda)
    target_compile_definitions(elips_gpu PRIVATE ELIPS_CUDA_ENABLED)
endif()
# ... same pattern for HIP, SYCL, Vulkan
```

Note the linkage difference:
- **Metal**: `PUBLIC` linkage — Metal types are accessible through `elips_gpu`. On Apple platforms, Metal is the primary backend.
- **CUDA/HIP/SYCL/Vulkan**: `PRIVATE` linkage — compile definitions are `PRIVATE`. Backend types are only visible within the GPU library.

### Metal Backend Build

```cmake
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

add_library(elips_gpu_metal STATIC MetalBackend.mm)

set_source_files_properties(MetalBackend.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc -O3 -ffast-math"
)
target_link_libraries(elips_gpu_metal PUBLIC
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)
```

- **Objective-C++**: `.mm` file compiled with `-fobjc-arc` (Automatic Reference Counting).
- **Optimization**: `-O3 -ffast-math` for maximum shader compilation and kernel performance.
- **Frameworks**: Metal, Metal Performance Shaders, Foundation.

## Build Configuration Summary

### Typical Build Commands

```bash
# Minimal build (CPU only, with tests)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# With Python bindings
cmake -B build -DCMAKE_BUILD_TYPE=Release -DELIPS_BUILD_PYTHON=ON
cmake --build build

# With CUDA backend
cmake -B build -DCMAKE_BUILD_TYPE=Release -DELIPS_GPU_CUDA=ON
cmake --build build

# Run tests
cd build && ctest --output-on-failure
```

### Compile Definitions

| Define | When Set | Effect |
|--------|----------|--------|
| `ELIPS_GPU_ENABLED` | Any GPU backend ON | Enables GPU types in core headers (`elips.hpp`, `Config.hpp`) |
| `ELIPS_METAL_ENABLED` | `APPLE AND ELIPS_GPU_METAL` | Enables Metal backend code paths |
| `ELIPS_CUDA_ENABLED` | `ELIPS_GPU_CUDA` | Enables CUDA backend code paths |
| `ELIPS_HIP_ENABLED` | `ELIPS_GPU_HIP` | Enables HIP/ROCm backend code paths |
| `ELIPS_SYCL_ENABLED` | `ELIPS_GPU_SYCL` | Enables SYCL backend code paths |
| `ELIPS_VULKAN_ENABLED` | `ELIPS_GPU_VULKAN` | Enables Vulkan backend code paths |

### SIMD Optimization (Automatic)

The `Metrics.cpp` source uses compile-time detection for SIMD kernels:

```cpp
#if defined(__ARM_NEON)
#include <arm_neon.h>
// NEON-accelerated dot product and squared L2 distance
// Runtime dispatch via static Dispatch singleton
#endif
```

On x86, the scalar fallback is used (AVX2/AVX-512 kernels are planned as future optimizations via CPUID dispatch).