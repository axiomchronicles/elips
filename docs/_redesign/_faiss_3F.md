# Phase 3F — FAISS Build System & Packaging Study

Source: FAISS v1.15.1 (this tree already carries a Metal backend + Dynamic
Dispatch, so it is a Metal-aware fork, not vanilla upstream). Read-only; nothing
was built. All citations are `file:line` relative to `/tmp/faiss`.

> **Reality-check header.** Three expectations from the task brief are WRONG
> against the real tree:
> - FAISS requires **C++20**, not C++23 (`CMakeLists.txt:83` sets
>   `CMAKE_CXX_STANDARD 20`; `INSTALL.md:82` says "a C++20 compiler"). The Metal
>   `.mm` sources are even `OBJCXX_STANDARD 17` (`faiss/gpu_metal/CMakeLists.txt:52`).
> - `FAISS_OPT_LEVEL` is **not** a `generic/avx2/avx512/avx512_spr/sve` enum only
>   — this fork adds a sixth value **`dd`** (runtime "Dynamic Dispatch") and it is
>   the DEFAULT for pip wheels (`pyproject.toml:59`). The default in CMake itself
>   is `generic` (`CMakeLists.txt:88`).
> - `FAISS_ENABLE_MKL` exists as a first-class option (`CMakeLists.txt:93`,
>   default `ON`) — MKL is not purely `BLA_VENDOR`-driven.
> - The brief's `demo_ivfpq_indexing_gpu` and `demo_auto_tune.py` are documented
>   in `INSTALL.md` but are NOT CMake targets in `demos/CMakeLists.txt` (see §5).

---

## 1. CMake OPTIONS — exact names, defaults, locations

All in top-level `CMakeLists.txt` unless noted. `cmake_minimum_required` is
**3.24.0** (`CMakeLists.txt:22`); `project(... VERSION 1.15.1 ...)` at
`CMakeLists.txt:76-80`.

| Option | Default | Line | Notes |
|---|---|---|---|
| `FAISS_OPT_LEVEL` | `"generic"` | `:88` | enum: `generic`/`avx2`/`avx512`/`avx512_spr`/`sve`/**`dd`**. `option()` abused as a string here. |
| `FAISS_ENABLE_GPU` | `ON` | `:89` | adds CUDA (or HIP) language. |
| `FAISS_GPU_STATIC` | `OFF` | `:90` | static-link CUDA (`faiss/gpu/CMakeLists.txt:173`). |
| `FAISS_ENABLE_CUVS` | `OFF` | `:91` | NVIDIA cuVS; needs `FAISS_ENABLE_GPU=ON`; pulls rapids-cmake (`:63-74`). |
| `FAISS_ENABLE_ROCM` | `OFF` | `:92` | AMD; swaps CUDA→HIP, runs `hipify.sh` (`:116`). |
| `FAISS_ENABLE_MKL` | `ON` | `:93` | gate for `find_package(MKL)` (`faiss/CMakeLists.txt:734`). |
| `FAISS_ENABLE_PYTHON` | `ON` | `:94` | SWIG bindings subdir. |
| `FAISS_ENABLE_C_API` | `OFF` | `:95` | C API subdir. |
| `FAISS_ENABLE_EXTRAS` | `ON` | `:96` | demos + benchs + tutorial/cpp. |
| `FAISS_USE_LTO` | `OFF` | `:97` | uses `check_ipo_supported` (`faiss/CMakeLists.txt:712`). |
| `FAISS_ENABLE_METAL` | `ON` on Apple-arm64 else `OFF` | `:98-104` | **fork-only.** Double-defaulted: pre-seeded before `project()` (`:48-61`) so `OBJCXX` is added as a language. `FATAL_ERROR` if set on non-Apple (`:138-140`). |
| `FAISS_ENABLE_SVS` | `OFF` | `:105` | Intel Scalable Vector Search; FetchContent-built. |
| `FAISS_SVS_RUNTIME_VERSION` | `"v0"` | `:106` | cache STRING. |
| `BUILD_TESTING` | `ON` (CTest default) | `:160-161` | `include(CTest)` then `add_subdirectory(tests)`. |
| `BUILD_SHARED_LIBS` | `OFF` (CMake default → `.a`) | `INSTALL.md:137,176` | standard CMake var, not redefined. |

GPU/BLAS tuning vars (not `option()`s, passed on CLI):
- `CMAKE_CUDA_ARCHITECTURES` — e.g. `"75;72"` (`INSTALL.md:160`); CI uses
  `"75-real;80;86-real;90;100;120"` (`.github/workflows/nightly.yml:30`).
- `CUDAToolkit_ROOT` — hint path (`INSTALL.md:157`).
- `BLA_VENDOR=Intel10_64_dyn` + `MKL_LIBRARIES=...` for MKL (`INSTALL.md:152`).
- `Python_EXECUTABLE` — pick a Python (`INSTALL.md:166`).

### Language selection (before `project()`)
`FAISS_LANGUAGES` starts `CXX` (`:24`), appends `CUDA` or `HIP` under GPU
(`:26-34`), appends `OBJCXX` under Metal (`:36-38`, `:59`). This ordering matters:
languages must be known at `project()` time, which is why Metal is pre-seeded
into the cache at `:48-61`.

## 2. Per-opt-level library targets — how they're produced

`faiss/CMakeLists.txt` builds the SAME `FAISS_SRC` list into FIVE static libs,
each with different SIMD flags + per-arch SIMD source files appended:

| Target | add_library | SIMD flags (non-Win) | Extra sources |
|---|---|---|---|
| `faiss` | `:452` | baseline / NEON mandatory on arm64 (`:629-632`) | NEON always on arm64; RVV on riscv (`:639`) |
| `faiss_avx2` | `:454` | `-mavx2 -mfma -mf16c -mpopcnt -mbmi2?` (`:459`) | `FAISS_SIMD_AVX2_SRC` (`:469`) |
| `faiss_avx512` | `:472` | +`-mavx512f/cd/vl/dq/bw` (`:479`) | AVX2+AVX512 src (`:485`) |
| `faiss_avx512_spr` | `:488` | +`vnni/vpopcntdq/bitalg/fp16/bf16` (`:495`) | AVX2+512+VPOPCNT+SPR (`:501`) |
| `faiss_sve` | `:512` | `-march=...+sve` detection (`:516-535`) | NEON+SVE (`:536`) |

Mechanism: a target is built only if it matches `FAISS_OPT_LEVEL`; otherwise
`EXCLUDE_FROM_ALL TRUE` (e.g. `:455-457`, `:473-475`). Higher levels INCLUDE
lower (`avx512` installs both `faiss_avx2` and `faiss_avx512`, `:792-798`). SIMD
file registry: `faiss/CMakeLists.txt:11-92`, split by arch via
`CMAKE_SYSTEM_PROCESSOR` match (`:80-92`). `-mbmi2` is feature-probed
(`:443-450`).

**`FAISS_OPT_LEVEL=dd` (Dynamic Dispatch, fork path):** instead of separate
libs, ALL SIMD sources are compiled INTO the main `faiss` target with per-file
compile options and `FAISS_ENABLE_DD` + `COMPILE_SIMD_*` defines
(`faiss/CMakeLists.txt:545-620`). Runtime picks the best kernel. The Python
loader selects the matching `swigfaiss_*` module at import
(`faiss/python/CMakeLists.txt:404` comment). This is how one pip wheel serves all
x86 CPUs.

Shared properties on all five: `POSITION_INDEPENDENT_CODE ON`,
`WINDOWS_EXPORT_ALL_SYMBOLS ON` (`:678-681`); `FINTEGER=int` (`:703-710`);
`OpenMP::OpenMP_CXX` linked PRIVATE (`:727-732`).

## 3. BLAS/LAPACK selection

`faiss/CMakeLists.txt:734-776`:
1. If `FAISS_ENABLE_MKL` → `find_package(MKL)`; if `MKL_FOUND`, link
   `${MKL_LIBRARIES}` to all 5 libs (`:737-742`).
2. Else user-provided `BLAS_LIBRARIES`/`LAPACK_LIBRARIES` win (`:744-746`).
3. Else prefer threaded OpenBLAS (`openblaso`/`openblasp`) and reuse it as
   LAPACK (`:751-759`) — avoids auditwheel double-bundling.
4. Else `find_package(BLAS REQUIRED)` + `find_package(LAPACK REQUIRED)`
   (`:761-763`).
On macOS arm64, pip builds use **Apple Accelerate** (found automatically,
`pyproject.toml:96` comment). conda osx uses OpenBLAS 0.3.34 (`conda/faiss/meta.yaml:59,72`).

## 4. Layered build FLOW (from INSTALL.md)

1. **Configure:** `cmake -B build .` (`INSTALL.md:120-122`).
2. **Build lib:** `make -C build -j faiss` → `libfaiss.a` (or `.so` if
   `-DBUILD_SHARED_LIBS=ON`) (`:173-177`). Build opt target BEFORE swig, e.g.
   `make -C build -j faiss_avx2` (`:185-201`).
3. **Python:** `make -C build -j swigfaiss` then
   `(cd build/faiss/python && python setup.py install)` (`:207-210`).
4. **Install C++:** `make -C build install` (`:216-224`).
5. **Test:** C++ `make -C build test` (needs `BUILD_TESTING=ON`, `:231-236`);
   Python `(cd build/faiss/python && python setup.py build)` then
   `PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py`
   (`:240-243`).

Install layout (`faiss/CMakeLists.txt:778-841`): `EXPORT faiss-targets`;
headers walked individually into `include/faiss/<dir>` (`:816-821`);
`faiss-config.cmake` + version file via `CMakePackageConfigHelpers` (`:823-841`)
→ this is the "mirror these conventions" target for ELIPS.

## 5. Demos & tests (reality vs brief)

**Demo targets actually defined** (`demos/CMakeLists.txt`, all
`EXCLUDE_FROM_ALL`): `demo_imi_flat`, `demo_imi_pq`, `demo_ivfpq_indexing` (`:12`),
`demo_nndescent`, `demo_sift1M` (`:18`), `demo_weighted_kmeans`,
`demo_residual_quantizer`, plus `add_subdirectory(diversity_filter)` (`:27`).
- `demo_ivfpq_indexing_gpu` — **not** in this CMakeLists (brief/INSTALL.md
  reference it; likely a GPU-subdir or stale doc). `INSTALL.md:264-267` still
  documents it.
- `demo_auto_tune.py` — a script run directly (`python demos/demo_auto_tune.py`,
  `INSTALL.md:296-317`), not a build target.

**C++ tests:** `tests/CMakeLists.txt` compiles `faiss_test` from `FAISS_TEST_SRC`
(`:6-49`), GoogleTest via FetchContent pinned to release-1.12.1 (`:114-122`),
discovered with `gtest_discover_tests` (`:162`). Run via CTest: `make -C build test`.
DD mode adds dispatch/perf tests (`:66-98`).

**Python tests:** conda runs
`python -X faulthandler -m unittest discover -s tests/ -p "test_*"` and
`"torch_*"` (`conda/faiss/meta.yaml:172-173`); pip smoke test is
`pytest tests/test_wheel_smoke.py` (`pyproject.toml:128`).

## 6. Packaging model

**conda (`conda/`, 3 recipes):**
- `conda/faiss/meta.yaml` → `libfaiss` (C++) + `faiss-cpu` (Python) subpackages,
  one recipe, per-platform build scripts (`:28-32`, `:99-103`). macOS-arm64 test
  asserts `libfaiss_metal.a` + `MetalDistance.metallib` exist (`:94-95`).
- `conda/faiss-gpu/meta.yaml` → GPU build, CUDA 12.6 or 13.2 pin logic
  (`:9-19`), x86_64-linux only.
- `conda/faiss-gpu-cuvs/meta.yaml` → separate cuVS package (CUDA 13.2, cuVS 26.06).
- Channels: `pytorch`, `nvidia`, `rapidsai`, `conda-forge` (`INSTALL.md:11-30`).
- Platform support (`INSTALL.md:7-9`): faiss-cpu on Linux x86-64/aarch64, OSX
  arm64 only, Windows x86-64; faiss-gpu Linux x86-64 only (CUDA 11.4/12.1);
  faiss-gpu-cuvs Linux x86-64 only.
- `conda_build_config.yaml`: Python 3.12/3.13/3.14, numpy 2.3.3.

**pip (`pyproject.toml`, scikit-build-core backend):**
- `name = "faiss-cpu"`, version regex-scraped from `CMakeLists.txt` (`:49-52`).
- Default CMake defines (`:58-68`): `FAISS_OPT_LEVEL=dd`, GPU/MKL/C_API/EXTRAS/
  SVS/LTO/TESTING all `OFF`, `BUILD_SHARED_LIBS=ON`.
- abi3 stable-ABI wheel: one `cp310` wheel covers 3.10+ (`:88-93`,
  `faiss/python/CMakeLists.txt:261-293` with `Py_LIMITED_API=0x030A0000`).
- Per-platform overrides: Linux/macOS LTO + dead-strip (`:71-86`); macOS points
  OpenMP at Homebrew libomp (`:97-112`); Windows → `generic` opt level (`:118`).
- cibuildwheel matrix (`:120-194`): cp310-cp314; repair via auditwheel/delocate/
  delvewheel; macOS `MACOSX_DEPLOYMENT_TARGET` 14.0 (arm64)/15.0 (intel).
- CI runners (`.github/workflows/build-pip.yml:42`): ubuntu-latest, macos-14,
  macos-15-intel, windows-2022, windows-11-arm, 2-core-ubuntu-arm.

**Nightlies:** `.github/workflows/nightly.yml` — scheduled `cron 10 6 * * *` +
`workflow_dispatch`; conda jobs push to `label: nightly` (`:21-24`) across
linux-x86_64 CPU, GPU (CUDA 13.2), and cuVS. conda nightly channel:
`pytorch/label/nightly` (`INSTALL.md:50-63`).

## 7. Compiler / dependency requirements

- **C++20** compiler with **OpenMP ≥ v2** (`INSTALL.md:82`, `CMakeLists.txt:83`).
  `find_package(OpenMP REQUIRED)` (`faiss/CMakeLists.txt:727`).
- BLAS impl, MKL recommended on Intel (`INSTALL.md:83-84`).
- Optional: nvcc + CUDA toolkit (GPU); ROCm (AMD); libcuvs=26.06 (cuVS);
  python3 + numpy + swig (bindings) (`INSTALL.md:86-97`).
- CMake ≥ 3.24.0 top-level; subdirs declare 3.17 (`faiss/python/CMakeLists.txt:6`,
  `c_api/CMakeLists.txt:6`). C API is C11 (`c_api/CMakeLists.txt:10`).
- Metal (fork): ObjC++17, `xcrun metal`/`metallib` to compile
  `MetalDistance.metal` → `.air` → `.metallib` at build time
  (`faiss/gpu_metal/CMakeLists.txt:56-77`); links `-framework Metal/MetalKit/
  MetalPerformanceShaders/Foundation` (`:29-37`).

---

## 8. ELIPS Phase-8 decisions — adopt / adapt / skip

| FAISS option / pattern | ELIPS mapping | Verdict |
|---|---|---|
| `FAISS_ENABLE_GPU` (umbrella) | `ELIPS_ENABLE_GPU` as umbrella that implies a backend | **adapt** — ELIPS is Metal-first; make it default `ON` only when a backend is viable, don't force CUDA language like FAISS does. |
| `FAISS_ENABLE_METAL` (default-on Apple-arm64, pre-seed OBJCXX, FATAL on non-Apple) | `ELIPS_ENABLE_METAL` | **adopt wholesale** — this fork already solved the OBJCXX-before-`project()` ordering (`CMakeLists.txt:48-61`) and the `xcrun`→`.metallib` custom-command chain. Copy both verbatim. |
| `FAISS_ENABLE_ROCM` / `FAISS_ENABLE_CUVS` | `ELIPS_ENABLE_ROCM` / (no cuVS equiv) | **adapt / skip** — keep `ELIPS_ENABLE_CUDA` + optional `ELIPS_ENABLE_ROCM`; skip cuVS/rapids-cmake machinery (huge dep, NVIDIA-only). |
| `FAISS_ENABLE_PYTHON` + SWIG | `ELIPS_ENABLE_PYTHON` + **pybind11** | **adapt** — same gate, but ELIPS uses pybind11 not SWIG, so drop the 6× `swigfaiss_*` duplication. Keep the abi3/`Py_LIMITED_API` stable-ABI trick (`faiss/python/CMakeLists.txt:261-293`) — big wheel-count win. |
| `FAISS_ENABLE_C_API` | `ELIPS_ENABLE_C_API` | **adopt** — same shape, default `OFF`, own subdir. Low cost. |
| `FAISS_OPT_LEVEL` multi-lib (`generic/avx2/avx512/avx512_spr/sve`) | `ELIPS_OPT_LEVEL generic/neon/avx2/avx512` | **adapt** — ELIPS Metal-first + arm64 primary: NEON is mandatory-baseline (adopt FAISS `:629-632`), make `avx2/avx512` the opt-in x86 tiers. Drop `avx512_spr`/`sve` unless needed. |
| `FAISS_OPT_LEVEL=dd` (Dynamic Dispatch, default for wheels) | `ELIPS_OPT_LEVEL=dd` | **adopt the concept, defer the impl** — one portable wheel per arch is the right distribution story, but DD is a large runtime-dispatch subsystem. Ship `generic`+per-tier libs first; add `dd` when wheel-matrix pain is real. `# ponytail: static per-tier libs now, add runtime dd when wheel count hurts`. |
| `FAISS_ENABLE_MKL` + `BLA_VENDOR` fallback chain | ELIPS BLAS detection | **adapt** — ELIPS on Apple → **Accelerate** (FAISS already does this for pip, `pyproject.toml:96`). Keep OpenBLAS fallback; skip MKL unless x86 perf matters. |
| `BUILD_TESTING` / `BUILD_SHARED_LIBS` / `FAISS_USE_LTO` | same names | **adopt verbatim** — standard CMake vars; FAISS's LTO-via-`check_ipo_supported` guard (`faiss/CMakeLists.txt:712-725`) is clean. |
| `FAISS_GPU_STATIC` | — | **skip** — CUDA-static-link niche; add only if a CUDA backend ships. |
| `FAISS_ENABLE_SVS` (FetchContent runtime) | — | **skip** — Intel-specific, not relevant. |
| Install/export (`faiss-targets`, `faiss-config.cmake`, version file) | `elips-targets` / `elips-config.cmake` | **adopt wholesale** — `CMakeLists.txt`(faiss)`:778-841` is the exact export convention to mirror; CMakePackageConfigHelpers + per-header install loop. |
| scikit-build-core pyproject + cibuildwheel matrix | ELIPS pyproject | **adopt** — scikit-build-core is the modern pybind11-compatible backend; copy the overrides structure, abi3 wheel, delocate/auditwheel repair, macOS deployment-target pins. |
| conda 3-recipe split (cpu/gpu/gpu-cuvs) | `elips` / `elips-metal`?/`elips-cuda` | **adapt / defer** — mirror the cpu-vs-gpu package split only once a GPU package exists; a single Metal+CPU macOS package is enough initially. |
| C++ standard | **C++23** (ELIPS requirement) | **diverge deliberately** — FAISS is C++20; ELIPS targets C++23. Set `CMAKE_CXX_STANDARD 23` and keep ObjC++ at 17 like FAISS's Metal libs. |
| CMake min version | 3.24 | **adopt** — matches FAISS top-level; 3.24+ needed for modern CUDA-arch + `CMAKE_<LANG>_STANDARD 23` support anyway. |
