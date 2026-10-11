# Installation

## Requirements

- **Python**: 3.11 or later
- **Compiler**: C++23 capable (Clang 17+, GCC 13+, or MSVC 2022+)
- **CMake**: 3.24 or later
- **Ninja**: Recommended build generator

## Platform Support

| Platform | Status |
|---|---|
| macOS (ARM64 / x86_64) | Full support — CPU (Stable) + Metal GPU backend (Beta, validated on Apple M4) |
| Linux (x86_64) | Full support — CPU (Stable) + CUDA/ROCm GPU backends (Experimental, not hardware-validated) |
| Windows (x86_64) | Supported (MSVC) |

The CPU engine is Stable. GPU support is optional and controlled at build time;
the Python package functions without any GPU backend — only CPU types are
imported when no GPU is configured.

## Option 1: CMake Source Build (Recommended)

Build from source when you need GPU support or are developing against a local checkout.

```bash
# Clone and build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DELIPS_ENABLE_PYTHON=ON
cmake --build build --target elips_pymodule -j
```

This produces `_core.cpython-<version>-<platform>.so` (or `.dylib` on macOS) inside the `bindings/python/elips/` package directory.

### GPU Backend Flags

Add one or more GPU backend flags to the CMake invocation:

| Flag | Backend | Maturity |
|---|---|---|
| `-DELIPS_ENABLE_METAL=ON` | Apple Metal (macOS) — on by default on Apple platforms | Beta (Apple M4) |
| `-DELIPS_ENABLE_CUDA=ON` | NVIDIA CUDA | Experimental (not hardware-validated) |
| `-DELIPS_ENABLE_ROCM=ON` | AMD ROCm/HIP | Experimental (not hardware-validated) |

The pre-Phase-5 flags `ELIPS_GPU_METAL` / `ELIPS_GPU_CUDA` / `ELIPS_GPU_HIP`
are honored as deprecated aliases (CMake emits a deprecation message).

```bash
cmake -S . -B build -G Ninja -DELIPS_ENABLE_PYTHON=ON -DELIPS_ENABLE_CUDA=ON
cmake --build build --target elips_pymodule -j
```

When any GPU backend is enabled, the `_core` extension is compiled with `ELIPS_GPU_ENABLED`, and GPU types (`GpuConfig`, `GpuDeviceInfo`, `GpuMetricsSnapshot`, etc.) become available in the Python package.

## Option 2: pip Install

`pip install` invokes the CMake build via `setup.py`:

```bash
pip install bindings/python/
```

Under the hood, `setup.py` runs:

```bash
cmake -S . -B <tmp> -DCMAKE_BUILD_TYPE=Release \
      -DELIPS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DELIPS_BUILD_CLI=OFF
cmake --build <tmp> --target elips_pymodule -j
```

The `pip` route does not expose GPU backend flags directly; use the CMake route if you need GPU acceleration.

## Option 3: PYTHONPATH (Development)

Set `PYTHONPATH` to point at the bindings directory to use the extension in-place without installation:

```bash
export PYTHONPATH=$PWD/bindings/python
```

Or inline:

```bash
PYTHONPATH=bindings/python python3 -c "import elips; print(elips.__version__)"
```

## Verifying Installation

```bash
python3 -c "import elips; print('version:', elips.__version__)"
```

Expected output:

```
version: 1.1.0
```

### Check GPU Availability

```python
import elips

if elips.has_gpu:
    print("GPU support enabled")
    print(elips.GpuPolicy.__members__)
else:
    print("CPU only")
```

`elips.has_gpu` reflects whether GPU bindings were compiled in — it is a build
flag, not runtime device detection. A build may report `has_gpu == True` yet
still fall back to CPU at runtime if no device is present.

### Check Package Files

The installed package should contain:

```
elips/
├── __init__.py        # Public API re-exports (~40 curated names)
├── _native.py         # The one seam onto the compiled extension
├── native.py          # Supported escape hatch to native handles
├── _core.pyi          # Type stubs (consumed by IDEs & type checkers)
├── _core.cpython-3XX-<arch>.so   # Compiled pybind11 extension
├── py.typed           # PEP 561 marker
├── exceptions.py      # Error hierarchy re-exports
├── typing.py          # Literals, aliases, and TypedDict models
├── collection/        # Collection: add/query/get/delete/...
├── config/            # connect() / connect_with_config()
├── database/          # Engine lifecycle wrapper
├── records/           # RecordInput / Row / Hit dataclasses
├── gpu/               # Accelerator wrapper
└── _internal/         # Embedder protocol, record coercion
```

The `_core` extension is the pybind11-compiled C++ module. The `.pyi` stub
provides static type information for it. The `py.typed` file signals to type
checkers that the package ships inline types. (The pre-split `modern.py` /
`core.py` facades remain only as deprecated import shims; there is no
`errors.py` or `types.py` — those are `exceptions.py` and `typing.py`.)

## Build Options Summary

| CMake Option | Default | Description |
|---|---|---|
| `ELIPS_ENABLE_PYTHON` | `OFF` | Build the Python bindings |
| `BUILD_TESTING` | `ON` | Build C++ test suite (turned OFF by pip) |
| `ELIPS_BUILD_CLI` | `ON` | Build CLI tool (turned OFF by pip) |
| `ELIPS_ENABLE_METAL` | `ON` (Apple) | Metal GPU backend (Beta) |
| `ELIPS_ENABLE_CUDA` | `OFF` | NVIDIA CUDA backend (Experimental) |
| `ELIPS_ENABLE_ROCM` | `OFF` | AMD ROCm/HIP backend (Experimental) |

The old `ELIPS_BUILD_PYTHON`, `ELIPS_BUILD_TESTS`, and `ELIPS_GPU_*` names are
honored as deprecated aliases.

## Troubleshooting

**ImportError: `_core` module not found**
: Ensure `bindings/python/elips/` is on `PYTHONPATH` and that `_core.so` was built (check for the file). Run `cmake --build build --target elips_pymodule` explicitly.

**ImportError: undefined symbol**
: The extension was linked against a different ABI than the Python interpreter. Rebuild with matching compiler and Python version.

**DimensionMismatch on first add**
: The vector you passed has a different length than the `dimension` argument to `Client()` / `PersistentClient()`.

**GPU types not available**
: The build was not configured with any GPU backend. Enable at least one backend (`-DELIPS_ENABLE_METAL=ON`, `-DELIPS_ENABLE_CUDA=ON`, or `-DELIPS_ENABLE_ROCM=ON`), which defines `ELIPS_GPU_ENABLED` for the extension.
