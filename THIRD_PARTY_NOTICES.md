# Third-Party Notices

ELIPS includes third-party material under the terms below.

Summary:

| Component | License | How included | Where it applies |
|---|---|---|---|
| FAISS (Meta Platforms) | MIT | algorithm adaptation (no source vendored) | Metal + CUDA + HIP GPU kernels |
| GoogleTest (Google) | BSD-3-Clause | fetched at build time (FetchContent), tests only | not shipped in library/wheel |
| pybind11 | BSD-3-Clause | fetched at build time (FetchContent), headers | Python bindings |
| NumPy | BSD-3-Clause | **runtime dependency, not vendored** | Index API numpy I/O |

## FAISS (GPU kernel algorithms)

ELIPS adapts GPU-kernel *algorithms* from FAISS. No FAISS source is copied or
vendored into this repository; the kernels below are reimplemented in ELIPS
style with ELIPS distance conventions, taking FAISS's designs as the precedent.

- **Metal.** The Apple Metal compute kernels in
  `src/gpu_engine/backends/metal/shaders/elips_distance_kernels.metal`
  (the tiled-GEMM distance-matrix kernels, the threadgroup top-k selection, and
  the IVF list-scan / list-merge kernels) and the
  `xcrun metal -> .air -> .metallib` build recipe in
  `src/gpu_engine/backends/metal/CMakeLists.txt` are adapted from FAISS
  (`faiss/gpu_metal/MetalDistance.metal` incl. `topk_threadgroup_2048`,
  `ivf_scan_list*`, `ivf_merge_lists`; `faiss/gpu_metal/CMakeLists.txt`).
- **CUDA (EXPERIMENTAL).** The on-device k-selection in
  `src/gpu_engine/backends/cuda/kernels/elips_cuda_kernels.cu` is adapted from
  FAISS's GPU block/warp-select design (`faiss/gpu/utils/Select.cuh`).
- **HIP / ROCm (EXPERIMENTAL).** The kernels in
  `src/gpu_engine/backends/hip/` mirror the CUDA path and follow FAISS's
  CUDA-like ROCm posture (hipBLAS SGEMM distance + device k-selection).

FAISS is licensed under the MIT License:

> Copyright (c) Meta Platforms, Inc. and affiliates.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## GoogleTest (Google) — BSD-3-Clause

The C++ test suite links GoogleTest v1.15.2, fetched at build time via CMake
`FetchContent` (`CMakeLists.txt`) and gated behind `BUILD_TESTING`. GoogleTest is
a **test-only** dependency: its install rules are suppressed
(`INSTALL_GTEST OFF`) and it is **not** shipped in the ELIPS library or Python
wheel. Licensed under the BSD 3-Clause License, Copyright 2008, Google Inc.

## pybind11 — BSD-3-Clause

The Python bindings (`bindings/python/elips_python.cpp`) are built with
pybind11 v2.13.6, fetched at build time via `FetchContent` (or a local source
tree via `ELIPS_PYBIND11_SOURCE_DIR`). pybind11 is header-only and used at
build time; no pybind11 source is vendored. Licensed under the BSD 3-Clause
License, Copyright (c) 2016 Wenzel Jakob.

## NumPy — BSD-3-Clause (runtime dependency, not vendored)

NumPy (`numpy>=1.21`) is a **runtime dependency** of the FAISS-style Index API
and `bf_knn` (float32 / C-contiguous array I/O). It is declared in `pyproject.toml`
/ `setup.py` and installed from PyPI by the end user; **no NumPy source is
vendored** in this repository. Licensed under the BSD 3-Clause License,
Copyright (c) 2005-2024, NumPy Developers.

---

### Build / runtime dependency summary

- **Vendored source:** none of the above is copied into this repository. FAISS
  contributes *algorithm designs only* (reimplemented); GoogleTest and pybind11
  are fetched at build time; NumPy is installed at runtime from PyPI.
- **Shipped in the ELIPS library / wheel:** none of these third-party libraries
  are redistributed as source or binary by ELIPS. NumPy is required at runtime
  but installed independently by the user's package manager.
