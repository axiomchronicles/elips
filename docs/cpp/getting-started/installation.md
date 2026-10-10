# Installation

## Compiler Requirements

ELIPS requires a C++23-capable compiler:

| Compiler | Minimum Version |
|----------|----------------|
| Clang    | 17+            |
| GCC      | 13+            |

C++23 features used: `std::flat_map`, `std::optional` monadic operations, `std::string_view`, `constexpr` enhancements, and `std::span`.

## Build System

CMake 3.24 or later is required. The top-level `CMakeLists.txt` sets `CMAKE_CXX_STANDARD 23` with `CMAKE_CXX_STANDARD_REQUIRED ON` and `CMAKE_CXX_EXTENSIONS OFF`.

## Building from Source

```bash
git clone https://github.com/anomalyco/elips.git
cd elips
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build . -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ELIPS_BUILD_CLI` | ON | Build the `elips` CLI tool |
| `ELIPS_BUILD_TESTS` | ON | Build the test suite (GoogleTest) |
| `ELIPS_BUILD_BENCH` | ON | Build the benchmark suite |
| `ELIPS_BUILD_PYTHON` | OFF | Build Python bindings (PyBind11) |
| `ELIPS_GPU_METAL` | ON | Apple Metal GPU backend |
| `ELIPS_GPU_CUDA` | OFF | NVIDIA CUDA GPU backend |
| `ELIPS_GPU_HIP` | OFF | AMD ROCm/HIP GPU backend |

Enabling any GPU backend automatically sets `ELIPS_GPU_ENABLED` and links the `elips_gpu` library. GPU backends are optional; the library works in pure-CPU mode without any GPU dependencies.

## Linking Against elips_core

The core library is built as `elips_core` (a static library). Its public include directory is `${CMAKE_CURRENT_SOURCE_DIR}/include`.

### CMake Integration via add_subdirectory

```cmake
add_subdirectory(path/to/elips)
target_link_libraries(my_app PRIVATE elips_core)
```

Targets linking `elips_core` automatically pick up the include path (`include/`) and the `ELIPS_GPU_ENABLED` compile definition when relevant.

### CMake Integration via find_package

Not yet available; use `add_subdirectory` or manually configure include paths and link the static library.

### Manual Linking

```bash
c++ -std=c++23 -I/path/to/elips/include my_app.cpp \
    /path/to/elips/build/libelips_core.a -o my_app
```

## Include Paths

The public API is organized under `include/elips/`. All public headers are accessible via:

```cpp
#include "elips/elips.hpp"                // umbrella header
#include "elips/Config.hpp"               // configuration types
#include "elips/domain/Vector.hpp"        // Vector domain type
#include "elips/domain/RecordID.hpp"      // RecordID (UUIDv7)
#include "elips/domain/Record.hpp"        // Record, Payload, MetaValue
#include "elips/domain/Errors.hpp"        // exception hierarchy
#include "elips/domain/SearchResult.hpp"  // SearchResult struct
#include "elips/metadata/Filter.hpp"      // metadata filtering
#include "elips/vector_engine/Metrics.hpp" // distance, requires_normalization
#include "elips/index_engine/IndexPort.hpp"      // abstract index interface
#include "elips/index_engine/ExactIndex.hpp"     // brute-force index
#include "elips/index_engine/HierarchicalGraphIndex.hpp" // HNSW index
#include "elips/index_engine/IndexFactory.hpp"  // make_index factory
#include "elips/query_engine/EQLLexer.hpp"      // EQL tokenizer
#include "elips/query_engine/EQLParser.hpp"     // EQL parser
#include "elips/query_engine/AST.hpp"           // EQL AST types
#include "elips/query_engine/QueryExecutor.hpp" // EQL executor
#include "elips/storage/WAL.hpp"               // write-ahead log
#include "elips/kernel/LockManager.hpp"        // advisory file lock
```

The umbrella header `elips/elips.hpp` includes the core domain types, Config, IndexPort, LockManager, and Filter. GPU headers are conditionally included when `ELIPS_GPU_ENABLED` is defined.

## Platform Notes

### macOS (ARM / Apple Silicon)

- Native ARM NEON SIMD is detected and dispatched automatically for distance computations (dot product and squared L2).
- The Metal GPU backend (`ELIPS_GPU_METAL=ON`, default) requires the Metal, MetalPerformanceShaders, and Foundation frameworks. These are found via `find_library` and linked automatically.
- Objective-C++ compilation is enabled for the Metal backend shader wrappers.

### Linux (x86-64 / ARM64)

- Scalar fallback kernels are used on x86-64. AVX2/AVX-512 dispatch is a planned future optimization.
- The `flock()` system call is used for the advisory file lock (`LOCK_EX | LOCK_NB`).
- On ARM64 Linux with NEON, the same NEON kernels dispatch as on macOS ARM.

### Dependencies

ELIPS has zero mandatory third-party library dependencies in CPU-only mode. Core dependencies (standard library only):

- C++23 standard library
- POSIX (for `flock`, `open`, `close`, `fcntl`)
- `<arm_neon.h>` on ARM platforms (detected automatically)

Optional test/build dependencies:

- GoogleTest v1.15.2 (fetched via CMake FetchContent)
- PyBind11 v2.13.6 (fetched via CMake FetchContent, only when `ELIPS_BUILD_PYTHON=ON`)

## Build Artifact

```
build/
├── libelips_core.a          # the core static library
├── elips                     # CLI tool (if ELIPS_BUILD_CLI=ON)
├── elips_bench               # benchmark binary (if ELIPS_BUILD_BENCH=ON)
└── bindings/python/elips/
    └── _core.so             # Python module (if ELIPS_BUILD_PYTHON=ON)
```