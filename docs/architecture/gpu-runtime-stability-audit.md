# GPU Runtime Stability Audit

Date: 2026-06-16

> **Historical audit (names drifted).** This is a dated record of a specific
> fix; the narrative is accurate for its date. In current code the names are
> `GpuBackend` (was `GpuPort`, header `GpuBackend.hpp`), `BackendSelector` (was
> `GpuSelector`), `Collection` (was `Vault`), and `add(..., id=...)` (was
> `place(...)`). The by-value `free_device(GpuBuffer)` fix described here is in
> the current `GpuBackend` interface. For authoritative GPU behavior, see the
> [GPU guide](../gpu/index.md).

## Scope

This audit covers the GPU-backed runtime paths exercised by:

- `elips.GpuDeviceInfo()`
- `db.gpu_info()`
- GPU-backed `Vault` creation and teardown
- GPU-backed replace/update flows (`place(..., id=existing_id)`)
- Python modern API merge flows built on top of repeated IDs

## Discovery And Runtime Call Graph

### Runtime device snapshot

`elips.GpuDeviceInfo()`

1. `bindings/python/elips_python.cpp`
2. `elips::gpu::GpuDeviceManager::runtime_device_info()`
3. `GpuDeviceManager::probe_all_devices()`
4. backend probe/init (`MetalBackend`, or other compiled backend)
5. `GpuSelector::select()`
6. selected `GpuPort::device_info()`

### Database GPU startup

`elips.open_with_config(...).gpu_info()`

1. `bindings/python/elips_python.cpp`
2. `elips::open()` in `src/Database.cpp`
3. `configure_gpu_backend()`
4. `GpuDeviceManager::probe_all_devices()`
5. `GpuSelector::select()`
6. `ElipsInstance::set_gpu_backend()`
7. `ElipsInstance::vault()`
8. `Vault(..., gpu_backend_.get())`
9. `make_index(...)`
10. `GpuBruteForceIndex` or `GpuHybridIndex`

### GPU-backed replace/update path

`Vault::place(..., id=existing_id)`

1. `Vault::place()` in `src/Database.cpp`
2. existing record path calls `index_->remove(record_id)`
3. GPU brute-force path calls `GpuBruteForceIndex::remove()`
4. `remove()` calls `sync_device_buffer_best_effort()`
5. `sync_device_buffer_best_effort()` calls `release_buffer()`
6. `release_buffer()` calls `GpuPort::free_device(...)`
7. insert path then rebuilds device buffer through `allocate_device()` + `upload()`

## Root Causes

### Root Cause 1: backend lifetime ended before vault teardown

Source evidence:

- `include/elips/elips.hpp`
- `src/Database.cpp`
- `src/gpu_engine/GpuBruteForceIndex.cpp`
- `src/gpu_engine/GpuGraphIndex.cpp`

`Vault` owns GPU indexes. GPU indexes hold non-owning references to `GpuPort`.
`ElipsInstance` previously declared `vaults_` before `gpu_backend_`, so normal
member destruction ran in this order:

1. `gpu_backend_` destroyed
2. `vaults_` destroyed
3. GPU index destructors called `backend_.free_device(...)` on dead backend state

Observed runtime evidence:

- unsandboxed Python repro ended with `SIGSEGV`
- stack traces pointed at `ElipsInstance::~ElipsInstance`

### Root Cause 2: `GpuPort::free_device(GpuBuffer&&)` did not consume ownership

Source evidence:

- `include/elips/gpu_engine/GpuPort.hpp`
- `src/gpu_engine/GpuBruteForceIndex.cpp`
- `src/gpu_engine/backends/metal/MetalBackend.mm`

The interface used `GpuBuffer&&`, but an rvalue-reference parameter only binds
to the caller object; it does not move from it by itself.

That meant code like:

- `backend_.free_device(std::move(database_buffer_));`

did **not** clear `database_buffer_`.

When `GpuBruteForceIndex::remove()` freed the last device buffer and then the
next insert immediately called `release_buffer()` again, ELIPS released the same
Metal buffer twice.

Observed runtime evidence:

- minimal GPU replace-key repro died before the second `place()` returned
- `NSZombieEnabled=YES` reported:
  `-[AGXG16GFamilyBuffer release]: message sent to deallocated instance`

## Fixes

### Lifetime fix

- moved `gpu_backend_` ahead of `vaults_` in `include/elips/elips.hpp`
- added explicit `vaults_.clear()` in `ElipsInstance::~ElipsInstance()`

Result:

- vaults and GPU indexes now release while the backend owner is still alive

### Ownership-transfer fix

- changed `GpuPort::free_device` from `GpuBuffer&&` to `GpuBuffer` by value
- updated backend overrides and test doubles accordingly

Result:

- `std::move(buffer)` now move-constructs the callee parameter
- caller-side `GpuBuffer` becomes empty before repeated cleanup paths run
- repeated free/rebuild flows no longer double-release backend handles

## Backend Impact Matrix

| Backend | Device reporting fix | Teardown lifetime fix | `free_device` ownership fix | Runtime validated |
| --- | --- | --- | --- | --- |
| Metal | Yes | Yes | Yes | Yes |
| CUDA | Yes | Yes | Yes | No, backend is stubbed/incomplete in this repo |
| HIP | Yes | Yes | Yes | No, backend is stubbed/incomplete in this repo |
| CPU fallback | Yes | N/A | N/A | Yes |

Notes:

- the lifetime fix applies to any backend used through `Vault` GPU indexes
- the `free_device` fix applies to every backend override because the contract
  bug lived in `GpuPort`, not in Metal-specific code
- only Metal was exercised end-to-end on this host

## Permanent Regression Coverage

### C++

- `GpuBackendLifecycleTest.repeated_selection_reports_non_empty_device_metadata`
- `GpuBackendIntegrationTest.repeated_allocation_transfer_cleanup_remains_stable`
- `GpuBackendIntegrationTest.brute_force_index_replace_same_id_is_stable`
- `GpuDatabaseLifecycleProbe`

### Python

- `test_gpu_database_teardown_subprocess`
- `test_gpu_modern_merge_replaces_existing_key_subprocess`

## Validation Performed

Host runtime: Apple Silicon M4, Metal backend visible outside the managed
sandbox.

Executed successfully:

- direct unsandboxed Python teardown repro
- direct unsandboxed Python replace-key repro
- `ctest --test-dir build --output-on-failure`
- `PYTHONPATH=bindings/python python3 tests/python/test_bindings.py`
- `PYTHONPATH=bindings/python python3 bindings/python/test.py --gpu-algorithm ivf_pq`

Observed results:

- `GpuDeviceInfo()` reports `Apple M4 / metal`
- `db.gpu_info()` reports `Apple M4 / metal`
- GPU-backed database teardown exits cleanly
- GPU-backed repeated-ID replace/merge exits cleanly
- full C++ suite passed: `163/163` with `GpuMemoryPoolTest.acquire_and_release`
  still intentionally skipped by the existing test
- full Python binding suite passed: `30/30`
- Python binding example script completed in one normal Python process without
  `os._exit()`

## Residual Gaps

- `GpuMetricsSnapshot` is still mostly placeholder data
- non-Metal backends remain source-level only in this repo and were not runtime
  validated on host hardware
- `GpuMemoryPoolTest.acquire_and_release` is still skipped by existing project
  policy and remains a separate follow-up item
