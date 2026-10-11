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
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The project version is sourced from a single `project(elips VERSION x.y.z)` line
in the root `CMakeLists.txt` (currently **1.1.0**); the Python package and
`elips.__version__` read the same line.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ELIPS_OPT_LEVEL` | `neon` on arm64, else `generic` | SIMD baseline for `elips_core`: `generic` (no ISA flags), `neon` (arm64 baseline), `avx2` (`-mavx2 -mfma -mf16c`), `avx512`. Single-tier build (no runtime dispatch yet). |
| `ELIPS_ENABLE_PYTHON` | OFF | Build Python bindings (PyBind11) |
| `ELIPS_ENABLE_C_API` | OFF | Build `elips_c`, the `extern "C"` shim over `index_factory`/add/search/free |
| `ELIPS_ENABLE_BLAS` | OFF | Reserved no-op (CPU distance is hand/NEON, not BLAS); warns if toggled on |
| `ELIPS_ENABLE_GPU` | — | Umbrella GPU switch (set automatically when a backend is enabled) |
| `ELIPS_ENABLE_METAL` | ON (Apple) | Apple Metal GPU backend |
| `ELIPS_ENABLE_CUDA` | OFF | NVIDIA CUDA GPU backend (experimental, build-from-source) |
| `ELIPS_ENABLE_ROCM` | OFF | AMD ROCm/HIP GPU backend (experimental, build-from-source) |
| `BUILD_TESTING` | ON | Build the test suite (GoogleTest) |
| `BUILD_SHARED_LIBS` | OFF | Build shared instead of static libraries |

Enabling any GPU backend sets `ELIPS_GPU_ENABLED` and links the `elips_gpu`
library (plus `elips_gpu_metal` on Apple). GPU backends are optional; the
library works in pure-CPU mode with no GPU dependencies.

> The pre-1.1 option names `ELIPS_GPU_METAL/CUDA/HIP`, `ELIPS_BUILD_PYTHON`, and
> `ELIPS_BUILD_TESTS` are honored as deprecated aliases (they emit a CMake
> deprecation message); use the names above.

## Linking Against elips (find_package)

An install produces a config package, so downstream projects consume ELIPS via
`find_package`:

```cmake
find_package(elips REQUIRED)
target_link_libraries(app PRIVATE elips::elips_core)
```

Every library on `elips_core`'s PUBLIC interface shares the export set:
`elips_gpu` (GPU builds) and `elips_gpu_metal` (Apple) are added automatically,
and the Metal `.metallib` installs beside the libraries. A smoke consumer under
`cmake/consumer_smoke/` builds out-of-tree against an install prefix.

### CMake Integration via add_subdirectory

```cmake
add_subdirectory(path/to/elips)
target_link_libraries(my_app PRIVATE elips::elips_core)
```

Targets linking `elips_core` automatically pick up the include path (`include/`)
and the `ELIPS_GPU_ENABLED` compile definition when relevant.

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
#include "elips/index_engine/Index.hpp"          // abstract index interface
#include "elips/index_engine/IndexFlat.hpp"      // brute-force / exact index
#include "elips/index_engine/IndexHNSW.hpp"      // HNSW index
#include "elips/index_engine/IndexIVFFlat.hpp"   // CPU IVF (flat postings)
#include "elips/index_engine/IndexIVFPQ.hpp"     // CPU IVF + product quantization
#include "elips/index_engine/IndexIVFScalarQuantizer.hpp" // CPU IVF + SQ8
#include "elips/index_engine/IndexFactory.hpp"  // make_index / index_factory
#include "elips/c_api/elips.h"                  // C API (ELIPS_ENABLE_C_API)
#include "elips/query_engine/EQLLexer.hpp"      // EQL tokenizer
#include "elips/query_engine/EQLParser.hpp"     // EQL parser
#include "elips/query_engine/AST.hpp"           // EQL AST types
#include "elips/query_engine/QueryExecutor.hpp" // EQL executor
#include "elips/storage/WAL.hpp"               // write-ahead log
#include "elips/kernel/LockManager.hpp"        // advisory file lock
```

The umbrella header `elips/elips.hpp` includes the core domain types, Config,
`Index`, LockManager, and Filter. GPU headers are conditionally included when
`ELIPS_GPU_ENABLED` is defined.

## Platform Notes

### macOS (ARM / Apple Silicon)

- Native ARM NEON SIMD is the arm64 baseline (`ELIPS_OPT_LEVEL=neon`) for distance computations (dot product and squared L2).
- The Metal GPU backend (`ELIPS_ENABLE_METAL=ON`, default on Apple) requires the Metal, MetalPerformanceShaders, and Foundation frameworks. These are found via `find_library` and linked automatically. Metal is the hardware-validated GPU backend.
- Objective-C++ compilation is enabled for the Metal backend shader wrappers.

### Linux (x86-64 / ARM64)

- On x86-64, `ELIPS_OPT_LEVEL=generic` builds with no ISA flags; `avx2`/`avx512` opt into those kernels at compile time (single-tier, no runtime dispatch yet).
- The `flock()` system call is used for the advisory file lock (`LOCK_EX | LOCK_NB`).
- On ARM64 Linux with NEON, the same NEON kernels dispatch as on macOS ARM.

### Dependencies

ELIPS has zero mandatory third-party library dependencies in CPU-only mode. Core dependencies (standard library only):

- C++23 standard library
- POSIX (for `flock`, `open`, `close`, `fcntl`)
- `<arm_neon.h>` on ARM platforms (detected automatically)

Optional test/build dependencies:

- GoogleTest v1.15.2 (fetched via CMake FetchContent)
- PyBind11 v2.13.6 (fetched via CMake FetchContent, only when `ELIPS_ENABLE_PYTHON=ON`)

## Build Artifacts

```
build/
├── libelips_core.a          # the core library (static by default)
├── libelips_gpu.a           # GPU engine (if a GPU backend is enabled)
├── libelips_gpu_metal.a     # Metal backend + .metallib (Apple, if enabled)
├── libelips_c.a             # C API shim (if ELIPS_ENABLE_C_API=ON)
└── bindings/python/elips/
    └── _core.so             # Python module (if ELIPS_ENABLE_PYTHON=ON)
```