# Package Structure

The `elips` Python package is a hybrid of pure-Python modules and a compiled C++ extension. This document describes each file's role and the re-export pattern.

## File Layout

```
bindings/python/
├── setup.py                           # pip / cibuildwheel entry point
└── elips/
    ├── __init__.py                    # Public API facade over _core + modern.py
    ├── _core.pyi                      # Type stubs (924 lines) — consumed by IDEs & type checkers
    ├── _core.cpython-3XX-<arch>.so    # Compiled pybind11 extension (macOS: .dylib, Linux: .so)
    ├── modern.py                      # Typed higher-level Engine/Arena wrapper
    ├── py.typed                       # PEP 561 marker — empty file, signals presence of inline types
    ├── errors.py                      # Error hierarchy re-exports (convenience submodule)
    └── types.py                       # Type alias re-exports (convenience submodule)
```

## `__init__.py` — The Public API

```python
from ._core import (
    Comparator,
    Config,
    ConfigError,
    Database,
    DimensionMismatch,
    Durability,
    ElipsError,
    Filter,
    GraphParams,
    IndexType,
    InvalidVector,
    LockConflict,
    Metric,
    NotFound,
    ParseError,
    Result,
    StorageError,
    Token,
    TokenKind,
    Transaction,
    TransactionVault,
    Vault,
    VaultInfo,
    distance,
    metric_from_string,
    metric_to_string,
    open,
    open_with_config,
    validate_eql,
    requires_normalization,
    tokenize_eql,
)
```

The `__init__.py` facade re-exports both the compiled `_core` surface and the
typed helpers from `modern.py` (`connect()`, `connect_with_config()`,
`Engine`, `Arena`, `Row`, `Hit`, and the `Embedder` protocol). The extension
may expose additional internal symbols, but only those explicitly re-exported
here are part of the public API.

### GPU Conditional Imports

GPU types import conditionally. If the extension was compiled without GPU support (no `ELIPS_GPU_ENABLED` define), the GPU classes are not present in `_core`:

```python
try:
    from ._core import (
        GpuConfig,
        GpuDeviceInfo,
        GpuError,
        GpuIndexAlgorithm,
        GpuIndexBuildParams,
        GpuMetricsSnapshot,
        GpuPolicy,
        GpuPrecision,
        GraphBuildAlgo,
        GraphIndexBuildParams,
        IndexBuildMode,
        IvfPqBuildParams,
        KernelTiming,
    )
    _has_gpu = True
except ImportError:
    _has_gpu = False
```

The `_has_gpu` flag is a module-level boolean that user code can check before accessing GPU types:

```python
if elips._has_gpu:
    gpu = elips.GpuConfig()
    gpu.policy = elips.GpuPolicy.prefer_gpu
    gpu.algorithm = elips.GpuIndexAlgorithm.ivf_flat
    gpu.ivf_pq_params.n_lists = 1024
else:
    print("CPU only")
```

This pattern avoids `AttributeError` when the extension module was compiled without GPU backends.

### The `__all__` Export List

```python
__all__ = [
    # factory
    "open",
    "open_with_config",
    # core classes
    "Database",
    "Vault",
    "VaultInfo",
    "Filter",
    "Result",
    "Config",
    "GraphParams",
    "Transaction",
    "TransactionVault",
    # enums
    "Metric",
    "IndexType",
    "Durability",
    "Comparator",
    # EQL
    "Token",
    "TokenKind",
    "validate_eql",
    "tokenize_eql",
    # utilities
    "distance",
    "requires_normalization",
    "metric_from_string",
    "metric_to_string",
    # GPU types
    "GpuConfig",
    "GpuDeviceInfo",
    "GpuError",
    "GpuIndexAlgorithm",
    "GpuIndexBuildParams",
    "GpuMetricsSnapshot",
    "GpuPolicy",
    "GpuPrecision",
    "GraphBuildAlgo",
    "GraphIndexBuildParams",
    "IndexBuildMode",
    "IvfPqBuildParams",
    "KernelTiming",
    # errors
    "ElipsError",
    "DimensionMismatch",
    "InvalidVector",
    "ConfigError",
    "NotFound",
    "StorageError",
    "LockConflict",
    "ParseError",
]
```

`__all__` controls what `from elips import *` exposes. GPU types are appended
only when the extension was built with GPU support, so CPU-only installs keep a
stable import surface without dangling GPU names.

## `_core.cpython-XXX.so` — The Compiled Extension

The native extension is built by pybind11 from `bindings/python/elips_python.cpp` and linked against `elips_core`. The target name is set in CMake:

```cmake
set_target_properties(elips_pymodule PROPERTIES
    OUTPUT_NAME _core
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/elips
)
```

The output filename follows Python's extension module naming convention:
- macOS: `_core.cpython-314-darwin.so`
- Linux: `_core.cpython-314-x86_64-linux-gnu.so`
- Windows: `_core.cpython-314-win_amd64.pyd`

## `_core.pyi` — Type Stubs

A 924-line type stub file that provides static type information for every
class, enum, and function exposed by the `_core` extension. This file is
consumed by IDEs (PyCharm, VSCode/Pylance) and type checkers (MyPy, Pyright) to
enable autocompletion, inline documentation, and type validation.

The stub is declared as package data in `setup.py`:

```python
package_data={"elips": ["py.typed", "_core.pyi"]},
```

For details on the stub contents, see [Type Stubs & IDE Support](../typing/type-stubs.md).

## `py.typed` — PEP 561 Marker

An empty file whose presence signals to type checkers that the `elips` package ships inline type information. Without this marker, MyPy and Pyright would ignore `_core.pyi` and fall back to the dynamic extension module (or `Any`).

## `errors.py` — Error Hierarchy Submodule

A convenience module that re-exports the exception classes from `_core`:

```python
from ._core import (
    ConfigError,
    DimensionMismatch,
    ElipsError,
    InvalidVector,
    LockConflict,
    NotFound,
    ParseError,
    StorageError,
)
```

Users can import errors either way:

```python
import elips
elips.DimensionMismatch          # via __init__.py re-export

from elips.errors import DimensionMismatch  # via submodule
```

## `types.py` — Type Alias Submodule

Provides reusable type aliases for IDE convenience — no runtime behavior:

```python
MetaValue = Union[bool, int, float, str]
Vector = Sequence[float]
PayloadLike = Mapping[str, MetaValue]
BatchRecord = Mapping[str, Union[Vector, PayloadLike, str, None]]
FetchResult = Mapping[str, Union[str, Vector, PayloadLike, None]]
ScanResult = Mapping[str, Union[str, PayloadLike]]
QueryBindings = Mapping[str, Vector]
RecordDict = Mapping[str, Union[str, Vector, PayloadLike, None]]
```

These aliases are duplicated in `_core.pyi` so they are available at the type level. The `types.py` module provides them at runtime for annotations and documentation.

## `setup.py` — Build & Distribution

```python
setup(
    name="elips",
    version="1.0.0",
    description="Embedded local vector database (SQLite for vectors)",
    packages=["elips"],
    package_data={"elips": ["py.typed", "_core.pyi"]},
    ext_modules=[CMakeExtension("elips._core")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    zip_safe=False,
)
```

The `CMakeBuild` custom command invokes CMake configure + build with `ELIPS_BUILD_PYTHON=ON`. The `zip_safe=False` flag is required because the extension is a shared library that cannot be loaded from a zip archive.

Key points:
- **Python 3.9+** is the minimum version
- The extension is always built from source (no pre-built wheels)
- `package_data` ensures `py.typed` and `_core.pyi` are included in distributions
- `pip` builds are limited to CPU; GPU backends must be configured via direct CMake
