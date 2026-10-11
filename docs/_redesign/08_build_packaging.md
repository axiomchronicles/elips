# Phase 8A — Build System & Packaging (FAISS-style)

Scope: CMake options, install/export (config package), Python packaging, and
the latent CMake-break fixes. Conventions mirror FAISS (see `_faiss_3F.md`).

## CMake options (root `CMakeLists.txt`)

| Option | Default | Effect |
|---|---|---|
| `ELIPS_OPT_LEVEL` | `neon` on arm64, else `generic` | SIMD baseline for `elips_core`. `generic` = no ISA flags; `neon` = arm64 baseline (no flag needed, NEON is mandatory); `avx2` = `-mavx2 -mfma -mf16c` (`/arch:AVX2` on MSVC); `avx512` = AVX2 + `-mavx512f/cd/vl/dq/bw` (`/arch:AVX512`). Single-tier build — no runtime dispatch yet. |
| `ELIPS_ENABLE_C_API` | `OFF` | Builds `elips_c`, a thin `extern "C"` shim over `index_factory`/add/search/free (`src/c_api/`, header `include/elips/c_api/elips.h`). |
| `ELIPS_ENABLE_BLAS` | `OFF` | Deliberate no-op. ELIPS CPU distance is hand/NEON, not BLAS — the knob is reserved for a future batched-gemm path. Warns if toggled on. |

## Install / export

`install(TARGETS ... EXPORT elipsTargets)` + `configure_package_config_file` +
`write_basic_package_version_file` produce an installed package so downstream:

```cmake
find_package(elips REQUIRED)
target_link_libraries(app PRIVATE elips::elips_core)
```

Every library on `elips_core`'s PUBLIC interface shares the export set:
`elips_gpu` (GPU builds) and `elips_gpu_metal` (Apple) are added automatically,
and the Metal `.metallib` installs beside the libs. Smoke consumer:
`cmake/consumer_smoke/` (build it out-of-tree against an install prefix).

## Version — single source of truth

`project(elips VERSION x.y.z)` in the root `CMakeLists.txt` is the ONLY place
the version lives:
- the Python extension receives it as the `ELIPS_VERSION` compile definition and
  `elips.__version__` reads it (no hand-synced string in `elips_python.cpp`);
- `bindings/python/setup.py::_read_version()` scrapes the same `project()` line
  (works in a dev checkout and in an sdist — `CMakeLists.txt` is vendored), and
  `pyproject.toml` declares `dynamic = ["version"]`.

This resolves the old drift (Python package reported 1.0.0 vs CMake 1.1.0); both
are now 1.1.0, sourced from one line.

## Python packaging

- `numpy>=1.21,<3` is now a declared runtime dependency (the Index API takes and
  returns float32 numpy arrays).
- `requires-python = ">=3.11"`.

### CPU / GPU wheel split rationale

- **macOS arm64** → ONE wheel ships CPU (NEON) + Metal. Metal is a system
  framework (no external toolkit), so bundling it costs nothing — no separate
  SKU, unlike FAISS's `faiss-cpu`/`faiss-gpu` split.
- **Linux / Windows x86_64** → CPU-only wheel (`generic`/`avx2` via
  `ELIPS_OPT_LEVEL`).
- **CUDA / HIP (ROCm)** → NOT shipped as wheels; they need a matching toolkit +
  driver on build and target hosts, so they are build-from-source
  (`-DELIPS_ENABLE_CUDA=ON` / `-DELIPS_ENABLE_ROCM=ON`).
- A single-tier wheel is built today; FAISS-style runtime dispatch (`dd`) is
  deferred until the x86 wheel matrix actually hurts.

## Latent CMake breaks fixed

1. `ELIPS_GPU_TESTS` added a non-existent `src/gpu_engine/tests` subdir →
   configure broke when toggled on. Now guarded (warns, GPU tests live in the
   top-level suite).
2. "ignoring duplicate libraries" (ld64): `elips_core` ↔ `elips_gpu` were
   mutually PUBLIC-linked (a static-lib cycle), and `elips_gpu_metal` was linked
   both directly and transitively. Fixed by dropping the `elips_gpu → elips_core`
   back-link (every binary links `elips_core` directly; ld64 resolves
   order-independently) and the redundant direct `elips_gpu_metal` links.
3. `install(EXPORT)` requires generator-expression include dirs → `elips_core`
   and `elips_gpu` PUBLIC includes now use `$<BUILD_INTERFACE>`/`<INSTALL_INTERFACE>`.

## Verified

`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DELIPS_ENABLE_PYTHON=ON`
→ build (no duplicate-library warnings) → `ctest` 307/307, 1 skip. C API:
`-DELIPS_ENABLE_C_API=ON` configures + `elips_c_demo` passes. Install +
`find_package` consumer links and runs. `test_bindings.py` 34/34;
`elips.__version__ == 1.1.0`.

## Follow-ups

- CUDA/HIP backend `CMakeLists` reference `.cu`/`.hip` sources that are actually
  `.cpp`; export include dirs there are untested (container-only). Fix when those
  toolchains are wired into CI.
- FAISS-style per-tier libs / runtime dispatch (`dd`) for a portable x86 wheel.
- abi3 / `Py_LIMITED_API` stable-ABI wheel to collapse the cp311–cp314 matrix.
