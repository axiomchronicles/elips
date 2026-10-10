# ============================================================================
# ELIPS CUDA backend  —  EXPERIMENTAL / NOT HARDWARE-VALIDATED
# Requires the NVIDIA CUDA Toolkit (nvcc) + cuBLAS. Built only when
# ELIPS_GPU_CUDA=ON (OFF by default). Compile-checked via
# scripts/ci/compile_cuda.sh; not validated on real NVIDIA hardware.
# ============================================================================

find_package(CUDAToolkit 12.0 QUIET)
if(NOT CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit not found; CUDA backend disabled")
    return()
endif()

enable_language(CUDA)

# Sane multi-arch default (Volta/Ampere/Hopper). sm_100 (Blackwell) omitted so
# this builds on CUDA 12.6; override with -DCMAKE_CUDA_ARCHITECTURES=... .
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
endif()
# The GpuBackend interface uses std::span (C++20). nvcc (CUDA 12.x) supports up to
# C++20 for device code, so pin to 20 here even though the host project is C++23;
# std::expected comes from the project's polyfill (domain/Expected.hpp).
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(elips_gpu_cuda STATIC
    CudaBackend.cu
    kernels/elips_cuda_kernels.cu
)

target_include_directories(elips_gpu_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
)

target_compile_definitions(elips_gpu_cuda PRIVATE ELIPS_CUDA_ENABLED)

target_link_libraries(elips_gpu_cuda PUBLIC
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(elips_gpu_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr>
)
