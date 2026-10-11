# Release Process

## Version Numbering

ELIPS follows **Semantic Versioning** (SemVer 2.0.0): `MAJOR.MINOR.PATCH`.

| Bump | When |
|---|---|
| MAJOR (x.0.0) | Breaking changes to the public C++ API, Python API, EQL syntax, storage format, or on-disk format |
| MINOR (0.x.0) | New features, new index types, new metrics, new GPU backends — all backward-compatible |
| PATCH (0.0.x) | Bug fixes, performance improvements, documentation updates — no API or format changes |

The current version is defined in `CMakeLists.txt`:

```cmake
project(elips LANGUAGES CXX VERSION 1.1.0)
```

The Python package version should match the C++ project version. It is exposed via `elips.__version__`.

## Build Types

Three build configurations are supported:

| Type | CMAKE_BUILD_TYPE | Flags | Purpose |
|---|---|---|---|
| Debug | `Debug` | `-O0 -g` | Development, debugging, sanitizers |
| Release | `Release` | `-O3 -DNDEBUG` | Production, benchmarks, distribution |
| RelWithDebInfo | `RelWithDebInfo` | `-O2 -g -DNDEBUG` | Profiling with optimizations (default) |

The default build type when not specified is `RelWithDebInfo`.

## Release Artifacts

A release should include:

### 1. C++ Library (`elips_core`)

- Static library built with `-DCMAKE_BUILD_TYPE=Release`
- Public headers under `include/elips/`
- Consumers link via CMake `target_link_libraries(... elips_core)` or manual include + link

### 2. CLI Binary (`elips`)

- Standalone executable built from `cli/src/main.cpp`
- Commands: `info`, `collections`, `stats`, `verify`, `query`, `checkpoint`, `quantize`, `export`, `import`, `bench`
- Distributed as a single binary (no runtime dependencies beyond the system libc)

### 3. Python Wheel

Build with:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DELIPS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DELIPS_BUILD_CLI=OFF
cmake --build build --target elips_pymodule -j
```

The built module goes to `bindings/python/elips/_core.<platform-ext>.so`. For packaging as a wheel, use a standard Python build backend (`setuptools`, `scikit-build-core`, etc.) with the CMake build as a subprocess.

The wheel should include:
- `bindings/python/elips/__init__.py`
- `bindings/python/elips/_core.*.so` (compiled extension)
- `bindings/python/elips/_core.pyi` (type stubs)
- `bindings/python/elips/py.typed` (PEP 561 marker)

## Pre-Release Checklist

### 1. Full test suite passes

```bash
# Debug build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure

# Release build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# Sanitizers
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### 2. Python bindings build and test

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DELIPS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DELIPS_BUILD_CLI=OFF
cmake --build build --target elips_pymodule -j
PYTHONPATH=bindings/python python3 -c "import elips; print(elips.__version__)"
PYTHONPATH=bindings/python python3 tests/python/test_bindings.py
```

### 3. Documentation is up to date

- [ ] C++ SDK docs (`docs/cpp_sdk.md`)
- [ ] Python SDK docs (`docs/python_sdk.md`)
- [ ] CLI docs (`docs/cli.md`)
- [ ] EQL language guide (`docs/eql.md`)
- [ ] Storage docs (`docs/storage.md`)
- [ ] Architecture docs (`docs/architecture.md`)
- [ ] Cookbook (`docs/cookbook.md`)
- [ ] API reference indexes (`docs/api/cpp/index.md`, `docs/api/python/index.md`)

### 4. No compiler warnings

Verify clean builds with `-Wall -Wextra -Wpedantic` (already set in CMakeLists.txt). Zero warnings in both Debug and Release on all supported platforms.

### 5. Version bump

Update the version in `CMakeLists.txt`:

```cmake
project(elips LANGUAGES CXX VERSION <new_version>)
```

Also update in:
- Python `__init__.py` (if version is hardcoded)
- Any version references in documentation

### 6. Tag and GitHub Release

```bash
git tag -a v<version> -m "ELIPS v<version>"
git push origin v<version>
```

Create a GitHub Release from the tag. Include:
- Release notes summarizing new features, fixes, and any breaking changes
- Links to updated documentation
- Binary artifacts if distributing pre-built binaries

## CMake Packaging

ELIPS can be consumed as a CMake package. The project structure supports:

```cmake
# In consumer's CMakeLists.txt
find_package(elips REQUIRED)
target_link_libraries(my_app PRIVATE elips::elips_core)
```

For this to work, ELIPS needs to be installed:

```bash
cmake --install build --prefix /path/to/install
```

Or consumed via `FetchContent`:

```cmake
FetchContent_Declare(
    elips
    GIT_REPOSITORY https://github.com/<org>/elips.git
    GIT_TAG v1.0.0
)
FetchContent_MakeAvailable(elips)
target_link_libraries(my_app PRIVATE elips_core)
```

## Post-Release

- [ ] Verify the GitHub Release page renders correctly
- [ ] Confirm that `FetchContent` consumers can pull the tagged version
- [ ] Verify `pip install` from the published wheel (if applicable)
- [ ] Update the roadmap (`docs/roadmap.md`) if this release completes planned items
- [ ] Create a new ADR if the release introduces significant architectural changes

## Hotfix Process

For critical bugs in a released version:

1. Branch from the release tag: `git checkout -b hotfix/v1.0.1 v1.0.0`
2. Apply the fix with tests
3. Bump the PATCH version
4. Merge to `main` and back to the release branch
5. Tag `v1.0.1` and create a new GitHub Release

The `main` branch should always reflect the latest released version plus any unreleased development.