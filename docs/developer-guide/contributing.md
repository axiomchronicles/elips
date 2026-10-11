# Contributing to ELIPS

## Prerequisites

- **CMake** >= 3.24
- **Ninja** build generator
- **C++23 compiler**: Clang 17+ or GCC 13+
- **Python 3.12+** (for Python bindings and tests)

## Building

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Build targets:

| Target | Description |
|---|---|
| `elips_core` | Static library |
| `elips` (output name of `elips_cli`) | Command-line tool |
| `elips_bench` | Benchmark suite |
| `elips_tests` | C++ test suite |
| `elips_pymodule` | Python bindings (`ELIPS_ENABLE_PYTHON=ON`) |

Build types: `Debug`, `Release`, `RelWithDebInfo` (default if unspecified).

GPU backends are enabled via CMake options:
- `ELIPS_ENABLE_METAL` (default ON on Apple platforms)
- `ELIPS_ENABLE_CUDA`, `ELIPS_ENABLE_ROCM` (both default OFF, experimental, not hardware-validated)

The pre-rename option names (`ELIPS_GPU_METAL/CUDA/HIP`, `ELIPS_BUILD_PYTHON`,
`ELIPS_BUILD_TESTS`) are honored as deprecated aliases with a `message(DEPRECATION)`.

## Running Tests

### All tests

```bash
ctest --test-dir build --output-on-failure
```

### Filtering with GoogleTest

Since tests are registered via `gtest_discover_tests`, use the CTest `-R` flag for regex matching or `--gtest_filter` directly:

```bash
# Run a specific test suite via CTest regex
ctest --test-dir build -R IndexFlatTest

# Run a specific test case
ctest --test-dir build -R "MetricsTest.CosineIdenticalNormalizedVectorsZeroDistance"

# Run all GPU tests
ctest --test-dir build -R "Gpu"

# Run all EQL tests
ctest --test-dir build -R "Eql"

# Run integration tests only
ctest --test-dir build -R "Integration"

# Run directly with gtest_filter for precise selection
./build/elips_tests --gtest_filter='IndexFlat*'
./build/elips_tests --gtest_filter='*Edge*'
./build/elips_tests --gtest_filter='FilterEdgeTest.MissingFieldDoesNotMatch'
```

### Python tests

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DELIPS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DELIPS_BUILD_CLI=OFF
cmake --build build --target elips_pymodule -j
PYTHONPATH=bindings/python python3 tests/python/test_bindings.py
```

### Sanitizer build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Code Style

ELIPS follows the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) (isocpp.github.io). Specifically:

- **C++23** standard with extensions disabled (`CMAKE_CXX_EXTENSIONS OFF`).
- Compiler flags: `-Wall -Wextra -Wpedantic` on all targets.
- RAII for all resource management (locks, files, memory).
- Value semantics by default; move-only or non-copyable where ownership is unique.
- Scoped enums (`enum class`) exclusively (Core Guidelines Enum.3).
- `[[nodiscard]]` on functions where ignoring the return value is a bug.
- Exceptions for error handling; no error codes, no `std::expected` in the public API.
- DIP (Dependency Inversion Principle): depend on interfaces (`Index`), never concretes.
- Header guards use `#ifndef`/`#define`/`#endif` with `ELIPS_` prefix.
- Snake_case for files, functions, and variables; PascalCase for classes, enums, and structs.
- Namespace: everything under `elips`.
- Internal detail helpers go in `elips::detail` namespace.

See [coding-standards.md](coding-standards.md) for the full coding conventions.

## Submitting Pull Requests

1. **Fork** the repository and create a feature branch from `main`.
2. **Keep commits focused**: each commit should be self-contained and pass CI.
3. **Add tests** for new functionality. Tests live in `tests/unit/`, `tests/integration/`, `tests/recovery/`, `tests/concurrency/`, or `tests/parity/`.
4. **Run the full test suite** locally before submitting:
   ```bash
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
   cmake --build build -j
   ctest --test-dir build --output-on-failure
   ```
5. **Update documentation** if your change affects the public API, CLI, or Python bindings.
6. **Open a PR** against `main` with a descriptive title and body. Link any related issues.

## Review Process

All PRs trigger two CI workflows:

### `build.yml` (on every push to `main` and every PR)

Runs across 4 combinations:
- `ubuntu-latest` / Debug
- `ubuntu-latest` / Release
- `macos-latest` / Debug
- `macos-latest` / Release

Each job: checkout → install Ninja → configure → build → `ctest --output-on-failure`.

Fail-fast is **disabled** so all 4 jobs complete independently.

### `test.yml` (on push to `main` only)

Two additional jobs:

1. **ASan + UBSan**: Ubuntu Debug build with `-fsanitize=address,undefined -fno-omit-frame-pointer`. All C++ tests run under sanitizers.

2. **Python bindings**: Ubuntu Release build with `ELIPS_ENABLE_PYTHON=ON`. Builds `elips_pymodule`, then runs `examples/python/01_getting_started.py` as a smoke test.

### What reviewers look for

- Does the code follow the C++ Core Guidelines?
- Are new public APIs `[[nodiscard]]` where appropriate?
- Are exceptions used for error handling, not return codes?
- Are new features covered by tests?
- Is the change additive (no breaking changes to public API without an ADR)?
- Thread safety for concurrent access patterns.
- No compiler warnings with `-Wall -Wextra -Wpedantic`.

## Project Structure

```
elips/
├── CMakeLists.txt          # Build system
├── include/elips/          # Public headers
│   ├── domain/             # Value types (Vector, RecordID, Errors)
│   ├── index_engine/       # Index, IndexFlat, IndexHNSW, IVF family
│   ├── kernel/             # LockManager
│   ├── metadata/           # Filter predicate engine
│   ├── query_engine/       # EQL lexer, parser, AST, executor
│   ├── storage/            # WAL, Serialization
│   ├── vector_engine/      # Metrics distance functions
│   └── gpu_engine/         # GPU backend interfaces and implementations
├── src/                    # Implementation files
│   └── gpu_engine/         # GPU backend implementations
├── tests/
│   ├── unit/               # Unit tests (gtest)
│   ├── integration/        # Integration tests
│   ├── recovery/           # WAL crash-recovery tests
│   ├── concurrency/        # Multi-thread tests
│   ├── parity/             # CPU/GPU parity tests
│   └── python/             # Python binding tests
├── cli/                    # Command-line tool source
├── bindings/python/        # PyBind11 bindings
├── benchmarks/             # Benchmark suite
├── docs/                   # Documentation
│   └── adr/                # Architecture Decision Records
└── examples/               # Usage examples
```

## Getting Help

- Check the [Architecture Decision Records](../adr/) for rationale behind design choices.
- Read the [coding standards](coding-standards.md) for conventions.
- See the [release process](release-process.md) for versioning and packaging.