# Installation

## Requirements

- **Python**: 3.9 or later
- **Compiler**: C++23 capable (Clang 17+, GCC 13+, or MSVC 2022+)
- **CMake**: 3.24 or later
- **Ninja**: Recommended build generator

## Platform Support

| Platform | Status |
|---|---|
| macOS (ARM64 / x86_64) | Full support — CPU + Metal GPU backend |
| Linux (x86_64) | Full support — CPU + CUDA/ROCm/SYCL/Vulkan GPU backends |
| Windows (x86_64) | Supported (MSVC) |

GPU support is optional and controlled at build time. The Python package functions without any GPU backend — only CPU types are imported when no GPU is configured.

## Option 1: CMake Source Build (Recommended)

Build from source when you need GPU support or are developing against a local checkout.

```bash
# Clone and build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DELIPS_BUILD_PYTHON=ON
cmake --build build --target elips_pymodule -j
```

This produces `_core.cpython-<version>-<platform>.so` (or `.dylib` on macOS) inside the `bindings/python/elips/` package directory.

### GPU Backend Flags

Add one or more GPU backend flags to the CMake invocation:

| Flag | Backend |
|---|---|
| `-DELIPS_GPU_METAL=ON` | Apple Metal (macOS) — on by default on Apple platforms |
| `-DELIPS_GPU_CUDA=ON` | NVIDIA CUDA |
| `-DELIPS_GPU_HIP=ON` | AMD ROCm/HIP |
| `-DELIPS_GPU_SYCL=ON` | Intel oneAPI/SYCL |
| `-DELIPS_GPU_VULKAN=ON` | Vulkan Compute |

```bash
cmake -S . -B build -G Ninja -DELIPS_BUILD_PYTHON=ON -DELIPS_GPU_CUDA=ON
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
      -DELIPS_BUILD_PYTHON=ON -DELIPS_BUILD_TESTS=OFF -DELIPS_BUILD_CLI=OFF
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
version: 1.0.0
```

### Check GPU Availability

```python
import elips

if elips._has_gpu:
    print("GPU support enabled")
    print(elips.GpuPolicy.__members__)
else:
    print("CPU only")
```

### Check Package Files

The installed package should contain:

```
elips/
├── __init__.py       # Public API re-exports
├── _core.pyi         # Type stubs (924 lines)
├── _core.so           # or _core.cpython-314-darwin.so, _core.cpython-314-x86_64-linux-gnu.so
├── modern.py         # Typed high-level wrapper
├── py.typed           # PEP 561 marker
├── errors.py          # Error hierarchy re-exports
└── types.py           # Type alias re-exports
```

The `_core` extension is the pybind11-compiled C++ module. The `.pyi` stub provides static type information for it. The `py.typed` file signals to type checkers that the package ships inline types.

## Build Options Summary

| CMake Option | Default | Description |
|---|---|---|
| `ELIPS_BUILD_PYTHON` | `OFF` | Build the Python bindings |
| `ELIPS_BUILD_TESTS` | `ON` | Build C++ test suite (turned OFF by pip) |
| `ELIPS_BUILD_CLI` | `ON` | Build CLI tool (turned OFF by pip) |
| `ELIPS_GPU_METAL` | `ON` (Apple) | Metal GPU backend |
| `ELIPS_GPU_CUDA` | `OFF` | NVIDIA CUDA backend |
| `ELIPS_GPU_HIP` | `OFF` | AMD ROCm backend |
| `ELIPS_GPU_SYCL` | `OFF` | Intel oneAPI backend |
| `ELIPS_GPU_VULKAN` | `OFF` | Vulkan compute backend |

## Troubleshooting

**ImportError: `_core` module not found**
: Ensure `bindings/python/elips/` is on `PYTHONPATH` and that `_core.so` was built (check for the file). Run `cmake --build build --target elips_pymodule` explicitly.

**ImportError: undefined symbol**
: The extension was linked against a different ABI than the Python interpreter. Rebuild with matching compiler and Python version.

**DimensionMismatch on first place**
: The vector you passed has a different length than the `dimension` argument to `open()`.

**GPU types not available**
: The build was not configured with any GPU backend. Set `ELIPS_GPU_ENABLED=ON` by enabling at least one backend (`ELIPS_GPU_METAL=ON`, `ELIPS_GPU_CUDA=ON`, etc.).
