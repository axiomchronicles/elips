# Contributing to ELIPS

First off, thank you for considering contributing to ELIPS! It's people like you that make ELIPS a great embedded vector database.

## Developer Guide

For detailed information on how to build, test, and contribute to ELIPS, please read our [Developer Guide](docs/developer-guide/contributing.md).

Here is a quick overview of what you need to know:

### Prerequisites
- **CMake** >= 3.24
- **Ninja** build generator
- **C++23 compiler**: Clang 17+ or GCC 13+
- **Python 3.12+** (for Python bindings and tests)

### Build & Test Flow
```bash
# Configure and Build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run C++ Tests
ctest --test-dir build --output-on-failure
```

### Python Bindings Development
```bash
# Build python bindings
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DELIPS_BUILD_PYTHON=ON -DELIPS_BUILD_TESTS=OFF -DELIPS_BUILD_CLI=OFF
cmake --build build --target elips_pymodule -j

# Run python tests
PYTHONPATH=bindings/python python3 tests/python/test_bindings.py
```

### Code Style
- We follow the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines).
- Python code is formatted and linted with **ruff**.
- Scoped enums, RAII, and value semantics are preferred.

## Code of Conduct

Please be respectful and professional when participating in the ELIPS community.

## Submitting Pull Requests

1. Fork the repository and create your branch from `main`.
2. Write clean C++23 / Python code and add tests for any new functionality.
3. Ensure all CI checks (linter, unit tests, sanitizers) pass.
4. Submit your pull request with a clear description of the changes.
