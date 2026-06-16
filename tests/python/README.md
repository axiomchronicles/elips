# ELIPS Python Test Suite

Validation tests for the ELIPS Python bindings in `bindings/python/elips`.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DELIPS_BUILD_PYTHON=ON -DELIPS_BUILD_TESTS=OFF -DELIPS_BUILD_CLI=OFF
cmake --build build --target elips_pymodule -j
```

## Run

```bash
PYTHONPATH=bindings/python python3 tests/python/test_bindings.py
PYTHONPATH=bindings/python python3 tests/python/test_local_embedder.py
```

The suite currently covers 26 binding and wrapper scenarios, including:

- exception and enum exposure
- utility helpers and EQL tooling
- config getters plus v2 config surface
- CRUD, filters, transactions, and context-manager behavior
- native document ingestion with `place_document()`
- `place_many()` for vector and text-first batches
- local/default embedder provisioning, persistence, and explicit-reopen rules
- planner output via `QueryPlan`
- modern `Engine` / `Arena` text, vector, and hybrid flows
- segmented persistence, `compact()`, `rebuild_index()`, and read-only reopen
- smoke-level leak, threading, and type-stub checks

GPU-specific tests skip automatically when no supported backend is available.
