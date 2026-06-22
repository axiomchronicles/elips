# ELIPS Release Process

This document describes how to release a new version of ELIPS. For a detailed guide on version numbering, build configurations, and packaging, see the [Release Process Guide](docs/developer-guide/release-process.md).

## Quick Release Steps

1. **Verify All Tests Pass Locally**:
   Ensure C++ tests, Python bindings, and sanitizers all pass:
   ```bash
   # Run sanitizers
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
   cmake --build build && ctest --test-dir build
   
   # Run Python bindings tests
   PYTHONPATH=bindings/python python3 tests/python/test_bindings.py
   ```

2. **Bump the Version**:
   - Update the version in `CMakeLists.txt`:
     ```cmake
     project(elips LANGUAGES CXX VERSION <new_version>)
     ```
   - Update the version in `bindings/python/pyproject.toml`.
   
3. **Commit the Bump**:
   ```bash
   git commit -am "chore: bump version to <new_version>"
   ```

4. **Tag the Release**:
   ```bash
   git tag -a v<new_version> -m "ELIPS v<new_version>"
   git push origin v<new_version>
   ```

5. **GitHub Actions Automation**:
   Pushing the tag triggers the release workflows:
   - Compiles and runs `ruff check` on Python code.
   - Builds C++ core and tests on Ubuntu & macOS.
   - Compiles cross-platform wheels for Python and publishes them to PyPI.
   - Generates release notes and creates a GitHub Release draft.
