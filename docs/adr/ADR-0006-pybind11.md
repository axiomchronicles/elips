# ADR-0006: PyBind11 for Python bindings

**Status:** Accepted
**Date:** 2024-08-01

## Context
The Python SDK must feel idiomatic, ship as a single wheel, and bind a C++23
API with value types, exceptions, and reference-lifetime semantics.

## Decision
Use PyBind11. It maps C++ exceptions to Python exceptions, handles STL
containers, supports `return_value_policy`/`keep_alive` for the
`Database → Vault` lifetime relationship, and builds a single extension module
(`elips._core`) re-exported by a thin `py.typed` package.

## Consequences
- Concise binding code; first-class type stubs (`_core.pyi`).
- Header-only dependency vendored via CMake `FetchContent`.
- Numpy zero-copy ingestion is a future enhancement (lists/iterables work today).

## Alternatives Considered
- **CPython C-API directly:** maximal control, far more boilerplate.
- **cffi/ctypes:** awkward for rich C++ objects and exceptions.
- **nanobind:** smaller/faster but less ubiquitous; revisit later.
