# Package Structure

The `elips` Python package is a hybrid of pure-Python modules and a compiled C++ extension. This document describes each file's role and the re-export pattern.

## File Layout

```
bindings/python/
├── setup.py                           # pip / cibuildwheel entry point
└── elips/
    ├── __init__.py                    # ~40 explicit exports; the public API
    ├── _native.py                     # The one seam onto the extension
    ├── native.py                      # Supported escape hatch to native handles
    ├── _core.pyi                      # Type stubs — consumed by IDEs & type checkers
    ├── _core.cpython-3XX-<arch>.so    # Compiled pybind11 extension
    ├── core.py                        # Deprecated alias for native.py
    ├── modern.py                      # Deprecated alias for the package root
    ├── collection/                    # Collection: writes, search, maintenance
    │   ├── collection.py              # The Collection implementation
    │   ├── _embedding.py              # Embedder fan-out, batched
    │   └── _hydration.py              # Native record dicts -> Row / Hit
    ├── config/connect.py              # connect() / connect_with_config()
    ├── database/engine.py             # Engine lifecycle wrapper
    ├── records/models.py              # RecordInput / Row / Hit dataclasses
    ├── search/                        # Filter, QueryPlan
    ├── query/                         # EQL: parse, tokenize, validate, AST
    ├── storage/                       # WAL replay
    ├── quantization/                  # Codec, QuantParams
    ├── gpu/accelerator.py             # Accelerator wrapper
    ├── _internal/
    │   ├── protocols.py               # Embedder protocol, typing helpers
    │   └── coercion.py                # Record normalization helpers
    ├── py.typed                       # PEP 561 marker
    ├── exceptions.py                  # Error hierarchy re-exports
    └── typing.py                      # Literals, aliases, and TypedDict models
```

## `__init__.py` — The Public API

The root exports ~40 curated names explicitly — no star imports. Native
handles (`Database`, `Vault`, `Filter`) resolve lazily through a module
`__getattr__` so existing code keeps working, but they are absent from
`__all__`: the supported spelling is `elips.native.Vault`.

Engine internals — the EQL AST, the lexer's `Token`, the index snapshots —
warn when reached through the root and name the module that owns them.

## `_native.py` — The Seam

Every module reaches the extension through `_native`, and nothing else
imports `_core`. The surface is derived rather than transcribed:

```python
globals().update({name: getattr(_core, name) for name in _public_names()})
```

`core.py` used to be a hand-maintained mirror of the extension's symbol
table, so adding one `py::class_` in C++ meant four mechanical Python-side
edits. A new binding now appears here the moment it compiles.

`_native` stays complete — it mirrors the extension, that is its job.
`supported_names()` is the curated subset that `elips.native` advertises;
`DEMOTED` maps each engine internal to the module that owns it now.

### GPU: A Conditional Surface

The GPU surface is conditional: built without GPU support (no
`ELIPS_GPU_ENABLED` define), the `Gpu*` classes are simply absent from
`_core`. The seam does not enumerate them in a `try`/`except` — it binds
whatever the extension exposes, so a conditional surface needs no
conditional import:

```python
has_gpu: bool = hasattr(_core, "GpuDevice")
```

`elips.has_gpu` is the public flag to check before touching GPU types:

```python
if elips.has_gpu:
    gpu = elips.native.GpuConfig()
    gpu.policy = elips.native.GpuPolicy.prefer_gpu
    gpu.algorithm = elips.native.GpuIndexAlgorithm.ivf_flat
    gpu.ivf_pq_params.n_lists = 1024
else:
    print("CPU only")
```

For most GPU work `elips.Accelerator` is the better entry point; the native
config types are the escape hatch for tuning it does not cover.

### The `__all__` Export List

The root `__all__` is a hand-curated list of ~40 names, grouped by task:
opening a database, collections and records, search, documents and
provenance, durability, compression, tuning, GPU, vector helpers, and
exceptions.

It is deliberately *not* derived from the extension. `_native` is generated
because mirroring a symbol table by hand is what drifted; the public API is
written down because deciding what belongs in it is a judgement call, and
the list is the place that judgement is recorded.

GPU names resolve through the same lazy `__getattr__`, so a CPU-only build
keeps a stable import surface without dangling GPU attributes.

## The High-Level API

The typed wrappers are the API. Each lives in the package named for what it
does:

- `config/connect.py` for high-level open/configure flows
- `database/engine.py` for the `Engine` lifecycle wrapper
- `collection/collection.py` for the typed write and query API
- `records/models.py` for `RecordInput`, `Row`, and `Hit`
- `_internal/protocols.py` for the `Embedder` protocol
- `_internal/coercion.py` for legacy-column to structured-record normalization

`Collection` was `Arena`, and `Arena` remains as an alias: `collection` is
the term comparable libraries use, and `arena` collides with the allocator
meaning in a project that also does memory management.

`RecordInput` is a typed dataclass grouping a record's vector, text,
document, metadata, and lineage into one reusable object. The older
column-oriented `ingest(...)` shape still works and normalizes into the
same structured path.

`collection/` is split three ways because the single-file version mixed
overload dispatch, embedder fan-out, and record coercion. `_embedding.py`
issues one embedder call per batch rather than one per record.

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
package_data = ({"elips": ["py.typed", "_core.pyi"]},)
```

For details on the stub contents, see [Type Stubs & IDE Support](../typing/type-stubs.md).

## `py.typed` — PEP 561 Marker

An empty file whose presence signals to type checkers that the `elips`
package ships inline type information. Without this marker, MyPy and
Pyright would ignore `_core.pyi` and fall back to the dynamic extension
module (or `Any`).

## The Deprecated Facades

`core.py` and `modern.py` are compatibility shims left over from the
pre-split layout, both emitting `DeprecationWarning` on import. They remain
for one minor version and nothing new should import them:

- `elips.core` re-exports the seam wholesale for code that imported the
  low-level API as `elips.core`.
- `elips.modern` re-exports the package root for code that imported the
  high-level API as `elips.modern`.

Both are lazy — they are in the root's `_LAZY` set, so they cost nothing
until touched, and the warning fires on import as intended.

## `exceptions.py` — Error Hierarchy Submodule

A convenience module that re-exports the exception classes from the seam:

```python
from elips._native import (
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

elips.DimensionMismatch  # exported from the package root

from elips.exceptions import DimensionMismatch  # via submodule
```

## `typing.py` — Runtime Typing Models

`typing.py` provides:

- primitive aliases such as `MetaValue`, `Vector`, and `PayloadLike`
- `Literal` names such as `MetricName`, `IndexName`, and `AccessModeName`
- `TypedDict` models such as `RecordInputDict`, `BatchRecord`, and `StoredRecord`

```python
MetaValue: TypeAlias = bool | int | float | str
Vector: TypeAlias = Sequence[float]
PayloadLike: TypeAlias = Mapping[str, MetaValue]
MetricName: TypeAlias = Literal["cosine", "euclidean", "dot_product"]
IndexName: TypeAlias = Literal["graph", "exact"]
AccessModeName: TypeAlias = Literal["read_write", "read_only"]
```

The primitive aliases are duplicated in `_core.pyi` so they are available at
the type level for the compiled extension. The richer `TypedDict` models only
exist in the pure-Python package.

`typing.py` imports native types under `TYPE_CHECKING` only. That guard is
load-bearing: the native modules import *this* module for their aliases, so
an unguarded import would close the loop into a real circular import at
module-execution time.

## `setup.py` — Build & Distribution

Metadata lives in `pyproject.toml` — name, version, `requires-python`, the
package list, and `package-data`. `setup.py` carries only what static
metadata cannot express:

```python
setup(
    ext_modules=[CMakeExtension("elips._core")],
    cmdclass={"build_ext": CMakeBuild, "sdist": VendorSourcesSdist},
    zip_safe=False,
)
```

`CMakeBuild` invokes CMake configure + build with `ELIPS_BUILD_PYTHON=ON`.
`zip_safe=False` is required because the extension is a shared library that
cannot be loaded from a zip archive.

`VendorSourcesSdist` copies the C++ sources into `core_src/` so an sdist can
compile. It runs in the `sdist` command — not at module scope. Populating it
on import meant that merely importing `setup.py`, which pip and most editors
do routinely, deleted and rewrote a directory tree, leaving an untracked
duplicate of the engine that silently went stale.

Key points:
- **Python 3.11+** is the minimum version
- `packages` in `pyproject.toml` lists the public subpackages
  (`elips.collection`, `elips.database`, `elips.query`, …) plus the private
  `elips._internal`
- The extension is always built from source (no pre-built wheels)
- `package-data` ensures `py.typed` and `_core.pyi` ship in distributions
- `pip` builds are limited to CPU; GPU backends must be configured via direct CMake
