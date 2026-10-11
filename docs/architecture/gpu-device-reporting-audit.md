# GPU Device Reporting Audit

Date: 2026-06-15

> **Historical audit (names drifted).** This is a dated record of a specific
> fix; the narrative is accurate for its date. In current code the names are
> `GpuBackend` (was `GpuPort`), `BackendSelector` (was `GpuSelector`),
> `Database` (was `ElipsInstance`), and `Collection` (was `Vault`). For the
> authoritative GPU behavior and maturity, see the [GPU guide](../gpu/index.md).

## Scope

This audit covers ELIPS GPU detection, runtime device reporting, Python bindings,
backend selection, and the validation work added to prevent regressions.

## Device Reporting Architecture

### Compile-time capability flag

`elips._has_gpu`

1. `bindings/python/elips/__init__.py`
2. Conditional import of GPU symbols from `elips._core`
3. `True` means GPU bindings were compiled into the extension module
4. `False` means the extension was built without GPU support

This flag is not runtime device detection.

### Runtime device snapshot

`elips.GpuDeviceInfo()`

1. Python constructor in `bindings/python/elips_python.cpp`
2. `elips::gpu::GpuDeviceManager::runtime_device_info()`
3. `GpuDeviceManager::probe_all_devices()`
4. Backend probe sequence:
   - Metal
   - CUDA
   - HIP
5. `GpuSelector::select()` chooses the preferred runtime backend
6. Returns `GpuPort::device_info()` when a backend initializes
7. Returns `GpuDeviceManager::cpu_fallback_info()` when no GPU backend is selected

### Database runtime snapshot

`db.gpu_info()`

1. Python binding in `bindings/python/elips_python.cpp`
2. `elips::ElipsInstance::gpu_info()`
3. Cached in `ElipsInstance::gpu_info_`
4. Populated during `configure_gpu_backend()` in `src/Database.cpp`
5. Uses:
   - `GpuDeviceManager::probe_all_devices()`
   - `GpuSelector::select()`
   - selected backend `device_info()`
6. Falls back to non-empty CPU metadata when no GPU backend is selected

## Root Cause

Two independent defects caused the empty metadata.

### Defect 1: Python constructor bound an empty value object

`bindings/python/elips_python.cpp` previously exposed `GpuDeviceInfo` with
`.def(py::init<>())`.

That created a zero-initialized `GpuDeviceInfo` struct and never queried any
runtime backend. As a result:

- `elips.GpuDeviceInfo()` always returned empty `name` and `backend`
- this happened even when a real GPU backend was available and working

### Defect 2: No fallback metadata when GPU backend selection failed

`src/Database.cpp::configure_gpu_backend()` previously set `gpu_info_` only
after successful device probe and backend selection.

When runtime probing failed or no backend was selected:

- `ElipsInstance::gpu_info_` remained the default zeroed struct
- `db.gpu_info()` returned empty `name`, `vendor`, and `backend`

This was especially visible in the managed sandbox because Metal enumeration was
blocked there.

### Additional architectural inconsistency

`configure_gpu_backend()` previously bypassed `GpuSelector` and called
`GpuDeviceManager::select()` directly. That meant database startup did not use
the same backend ranking rules as the rest of the GPU stack.

The fix now routes database selection through `GpuSelector`.

## Environment Findings

On this Apple Silicon host:

- sandboxed Metal probe:
  - `MTLCopyAllDevices()` returned `0`
  - `MTLCreateSystemDefaultDevice()` returned `nil`
- unsandboxed Metal probe:
  - reported `Apple M4`
  - Metal backend initialized successfully

This means the Metal backend itself was not broken on the host. The managed
sandbox hid the GPU from the process, which exposed the missing CPU-fallback
reporting path.

## Fix Implemented

### Production code

- Added `GpuDeviceManager::cpu_fallback_info()`
- Added `GpuDeviceManager::runtime_device_info()`
- Changed Python `GpuDeviceInfo()` to query runtime state instead of returning a
  zeroed struct
- Updated `configure_gpu_backend()` to initialize `gpu_info_` with CPU fallback
  before GPU selection
- Updated `configure_gpu_backend()` to use `GpuSelector`

### Contract after the fix

- `_has_gpu == True` means GPU bindings are compiled in
- `GpuDeviceInfo()` always returns non-empty device metadata
- `db.gpu_info()` always returns non-empty device metadata
- when a GPU backend is active, metadata reflects the selected GPU
- when no GPU backend is active, metadata reports CPU fallback

## Impact Matrix

| Surface | Status | Notes |
| --- | --- | --- |
| Python `GpuDeviceInfo()` | Fixed | Now returns runtime device snapshot |
| Python `db.gpu_info()` | Fixed | Now never returns zeroed metadata |
| Database startup selection | Improved | Now uses `GpuSelector` |
| Metal backend | Working | Verified on Apple M4 outside sandbox |
| CUDA backend | Stubbed/incomplete | Source exists but backend is not production-ready |
| HIP backend | Stubbed/incomplete | Source exists but backend is not production-ready |

## Validation Summary

### Sandboxed validation

- `cmake --build build -j4`
- `ctest --test-dir build -R "GpuDeviceManagerTest|GpuBackendIntegrationTest|CpuGpuRecallParityTest" --output-on-failure`
- `PYTHONPATH=bindings/python python3 tests/python/test_bindings.py`

Results:

- C++ regression tests passed
- Python binding suite passed
- GPU integration tests skipped when Metal was not visible
- `GpuDeviceInfo()` and `db.gpu_info()` both reported `CPU (SIMD) / cpu`

### Unsandboxed host validation

- standalone Metal probe reported `Apple M4`
- `ctest --test-dir build -R "GpuDeviceManagerTest|GpuMemoryManagerTest|GpuBackendLifecycleTest|GpuBackendIntegrationTest|CpuGpuRecallParityTest" --output-on-failure`
- `PYTHONPATH=bindings/python python3 tests/python/test_bindings.py`

Results:

- all targeted C++ GPU tests passed
- Python binding suite passed
- `GpuDeviceInfo()` and `db.gpu_info()` both reported `Apple M4 / metal`
- Metal allocation, transfer, distance, brute-force search, lifecycle, memory,
  and CPU-vs-GPU parity tests passed

## Regression Coverage Added

### C++

- `GpuDeviceManagerTest.cpu_fallback_info_is_non_empty`
- `GpuDeviceManagerTest.runtime_device_info_is_non_empty`
- `GpuDeviceManagerTest.database_gpu_info_matches_runtime_device_info`
- `GpuDeviceManagerTest.cpu_only_policy_reports_cpu_fallback_metadata`
- `GpuBackendLifecycleTest.repeated_selection_reports_non_empty_device_metadata`
- `GpuBackendIntegrationTest.repeated_allocation_transfer_cleanup_remains_stable`

### Python

- `test_gpu_device_info`
- `test_database_gpu_info_matches_runtime_snapshot`
- `test_cpu_only_gpu_policy_reports_cpu_fallback`
- updated `test_modern_probe_hybrid_and_explain` to assert against actual runtime
  backend selection instead of a sandbox-only CPU assumption

## Residual Gaps

- GPU metrics reporting (`gpu_stats`) is still largely unimplemented
- Non-Metal backends remain stubbed and were not validated as production paths
- The repo does not yet have a large dedicated `tests/gpu/...` harness; current
  coverage extends the existing unit, integration, parity, and Python binding
  suites
