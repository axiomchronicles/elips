# ELIPS Python Test Suite

Validation tests for the ELIPS Python bindings (`bindings/python/elips`).

## Dependencies

- Python 3.12+
- Built ELIPS Python module (`_core.*.so` under `bindings/python/elips/`)
- `py.typed` and `_core.pyi` type stubs (for the `test_type_stubs` test)

No third-party Python packages are required (the test file uses only the standard library: `sys`, `os`, `gc`, `threading`, `tempfile`, `traceback`).

## Building the Python Module

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DELIPS_BUILD_PYTHON=ON -DELIPS_BUILD_TESTS=OFF -DELIPS_BUILD_CLI=OFF
cmake --build build --target elips_pymodule -j
```

## Running Tests

```bash
PYTHONPATH=bindings/python python3 tests/python/test_bindings.py
```

Expected output:
```
  PASS test_exceptions
  PASS test_enums
  PASS test_utilities
  PASS test_eql_tooling
  PASS test_config
  PASS test_database_crud
  PASS test_place_many
  PASS test_filtered_search
  PASS test_transaction
  PASS test_database_context_manager
  PASS test_eql_query
  PASS test_gpu_config
  PASS test_gpu_device_info
  PASS test_thread_safety_python
  PASS test_edge_cases
  PASS test_memory_leak_check
  PASS test_type_stubs
  PASS test_parity_cpp_vs_python

Results: 18/18 passed, 0 failed
```

On systems without GPU support, `test_gpu_config` and `test_gpu_device_info` will show `SKIP` instead of `PASS`.

## Test Functions

### `test_exceptions()`
Verifies all 8 exception types are importable, are `Exception` subclasses, have the correct inheritance hierarchy (`DimensionMismatch` < `ElipsError`, `ParseError` < `ElipsError`, `LockConflict` < `ElipsError`), and are properly raised by the C++ layer (`DimensionMismatch` on wrong-size vector, `ParseError` on malformed EQL).

### `test_enums()`
Verifies all core enums (`Metric`, `IndexType`, `Durability`, `Comparator`) have distinct integer values and all GPU enums (`GpuPolicy`, `GpuError`, `GpuPrecision`, `GraphBuildAlgo`) exist (conditional on `_has_gpu`).

### `test_utilities()`
Tests `distance()` with both string and enum metric arguments, `requires_normalization()` for all three metrics, `metric_to_string()` and `metric_from_string()` round-trips.

### `test_eql_tooling()`
Tests `tokenize_eql()` output correctness (first token is "seek", TokenKind.word), `validate_eql()` succeeds on valid EQL and raises `ParseError` on invalid input.

### `test_config()`
Tests fluent Config builder: chain `.dimension().metric().index().durability().graph_params()`, verify `.dimension_val`, `.metric_val`, `.index_val` string getters, `.metric_enum`, `.index_enum`, `.durability_enum` enum getters, `.graph_params_val` accessor, and `open_with_config()`.

### `test_database_crud()`
Full CRUD lifecycle: `open()`, `vault()`, `place()` with metadata, `seek()` ordered by distance, `fetch()` existing and missing IDs, `scan()`, `erase()`, `count()`, `info()` with count/dimension, `list_vaults()`.

### `test_place_many()`
Batch insertion with `place_many()` using list of dicts `{"vector": [...], "data": {...}}`, with and without explicit UUID IDs.

### `test_filtered_search()`
Filter builder tests: chained `.field().equals().field().gte()` (AND), `.or_()` combinator, `.not_()` combinator, `.one_of()` set membership, `.contains()` substring match. Each filter is used with `vault.seek()` to verify correct result counts.

### `test_transaction()`
Transaction lifecycle: context manager commit (writes visible after `with` block), context manager rollback on exception (writes discarded), manual `commit()`, manual `rollback()`.

### `test_database_context_manager()`
Context manager `with elips.open(...)` on both in-memory and persistent databases. Verifies that the database is checkpointed on clean context exit (data survives reopen) and that `close()` is called.

### `test_eql_query()`
EQL execution via `db.query()`: place with data, seek with inline vector, fetch by id, scan with where/limit, erase by id. Verifies result counts and correctness.

### `test_gpu_config()`
GPU config property setters: `GpuConfig().policy`, `.algorithm`, `.device_memory_pool_mb`, `.fp16_search`, `.max_batch_size`, `.ef_search`. Also tests `GraphIndexBuildParams` and `IvfPqBuildParams`. Skipped if `_has_gpu` is False.

### `test_gpu_device_info()`
Verifies `GpuDeviceInfo` has all expected attributes: `name`, `peak_tflops_fp32`, `peak_tflops_fp16`, `supports_dynamic_batching`, `supports_half_precision_search`, `supports_bf16`, `compute_capability_major`, `host_to_device_bandwidth_gb_s`. Skipped if `_has_gpu` is False.

### `test_thread_safety_python()`
8 threads concurrently execute `seek()` on the same vault with 100 vectors. Each thread runs one query; all should get 10 results with no exceptions.

### `test_edge_cases()`
Boundary conditions: empty vector insertion (raises `DimensionMismatch`), 1000-dimensional vector, zero top-k returns ≤1 result, negative scan limit treated as "all", Unicode metadata (`"🚀"`, `"日本語"`) round-trips correctly.

### `test_memory_leak_check()`
50 cycles of: open database → get vault → insert 20 vectors → seek → scan → EQL query → delete database and vault. Runs `gc.collect()` at the end. Tests for leaks, not exact memory measurements.

### `test_type_stubs()`
Verifies `py.typed` marker file and `_core.pyi` stub file exist in the package directory. Ensures PEP 561 compliance for IDEs and type checkers.

### `test_parity_cpp_vs_python()`
End-to-end behavioral parity: places 4 vectors with metadata, verifies search returns expected nearest neighbor, fetch returns correct vector length, erase + fetch returns None, count decrements.

## CI Integration

The Python test suite is not directly run in CI (the `.github/workflows/test.yml` runs a simpler smoke test: `examples/python/01_getting_started.py`). To add the full test suite to CI, submit a PR that adds:

```yaml
- name: Test Python bindings
  run: PYTHONPATH=bindings/python python3 tests/python/test_bindings.py
```

to the Python bindings job in `.github/workflows/test.yml`.

## Writing New Python Tests

Follow the existing pattern:

1. Add a new `test_<feature>()` function
2. Add it to the `tests` list at the bottom of the file
3. Use assertions (`assert`) rather than pytest-style helpers
4. Print `"  PASS test_<name>"` on success
5. Clean up any filesystem resources (use `tempfile.TemporaryDirectory` for persistent db tests)
6. For GPU-dependent tests, skip gracefully with `if not elips._has_gpu: print("  SKIP ..."); return`