# HIP/ROCm backend — requires AMD ROCm SDK (hipcc + hipBLAS/rocBLAS).
#
# EXPERIMENTAL — NOT HARDWARE VALIDATED. Build from source against a ROCm
# install; there is no prebuilt package and no AMD GPU in CI (compile-checked
# only, mirroring FAISS's build-from-source ROCm posture). Enabled via
# -DELIPS_GPU_HIP=ON (alias: -DELIPS_ENABLE_ROCM=ON), OFF by default.

find_package(hip QUIET)
if(NOT hip_FOUND)
    message(STATUS "ROCm/HIP not found; HIP backend disabled")
    return()
endif()

# hipBLAS is the portable BLAS layer (backed by rocBLAS on AMD).
find_package(hipblas QUIET)
if(NOT hipblas_FOUND)
    message(STATUS "hipBLAS not found; HIP backend disabled")
    return()
endif()

enable_language(HIP)

add_library(elips_gpu_hip STATIC
    HipBackend.hip
    kernels/cosine_fp32.hip
)

# .hip files are compiled as HIP translation units (hipcc / amdgpu).
set_source_files_properties(HipBackend.hip kernels/cosine_fp32.hip
    PROPERTIES LANGUAGE HIP)

target_include_directories(elips_gpu_hip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
)

target_compile_definitions(elips_gpu_hip PUBLIC ELIPS_HIP_ENABLED)

target_link_libraries(elips_gpu_hip PUBLIC
    hip::host
    roc::hipblas
)

target_compile_options(elips_gpu_hip PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math>
)
