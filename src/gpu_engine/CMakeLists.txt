# ============================================================================
# ELIPS GPU Engine — CMake Build Module
# ============================================================================

add_library(elips_gpu STATIC
    GpuDeviceManager.cpp
    GpuSelector.cpp
    GpuIndexTransferManager.cpp
    GpuMemoryManager.cpp
    GpuMemoryPool.cpp
    DynamicBatcher.cpp
    GpuIngestionPipeline.cpp
    GpuSearchPipeline.cpp
    GpuQuantizationPipeline.cpp
    GpuProfiler.cpp
    detail/IvfIndexState.cpp
    GpuDistributedIndex.cpp
    GpuGraphIndex.cpp
    GpuIVFFlatIndex.cpp
    GpuIVFPQIndex.cpp
    GpuBruteForceIndex.cpp
    GpuHybridIndex.cpp
)

target_include_directories(elips_gpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/hip
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/sycl
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan
)

target_link_libraries(elips_gpu PUBLIC elips_core)
target_compile_options(elips_gpu PRIVATE -Wall -Wextra -Wpedantic)

# ---- CUDA backend ----
option(ELIPS_GPU_CUDA "Enable NVIDIA CUDA backend" OFF)
if(ELIPS_GPU_CUDA)
    add_subdirectory(backends/cuda)
    target_link_libraries(elips_gpu PRIVATE elips_gpu_cuda)
    target_compile_definitions(elips_gpu PRIVATE ELIPS_CUDA_ENABLED)
endif()

# ---- HIP backend ----
option(ELIPS_GPU_HIP "Enable AMD ROCm/HIP backend" OFF)
if(ELIPS_GPU_HIP)
    add_subdirectory(backends/hip)
    target_link_libraries(elips_gpu PRIVATE elips_gpu_hip)
    target_compile_definitions(elips_gpu PRIVATE ELIPS_HIP_ENABLED)
endif()

# ---- Metal backend ----
option(ELIPS_GPU_METAL "Enable Apple Metal backend" ON)
if(APPLE AND ELIPS_GPU_METAL)
    add_subdirectory(backends/metal)
    target_link_libraries(elips_gpu PUBLIC elips_gpu_metal)
    target_compile_definitions(elips_gpu PUBLIC ELIPS_METAL_ENABLED)
endif()

# ---- SYCL backend ----
option(ELIPS_GPU_SYCL "Enable Intel oneAPI/SYCL backend" OFF)
if(ELIPS_GPU_SYCL)
    add_subdirectory(backends/sycl)
    target_link_libraries(elips_gpu PRIVATE elips_gpu_sycl)
    target_compile_definitions(elips_gpu PRIVATE ELIPS_SYCL_ENABLED)
endif()

# ---- Vulkan backend ----
option(ELIPS_GPU_VULKAN "Enable Vulkan compute backend" OFF)
if(ELIPS_GPU_VULKAN)
    add_subdirectory(backends/vulkan)
    target_link_libraries(elips_gpu PRIVATE elips_gpu_vulkan)
    target_compile_definitions(elips_gpu PRIVATE ELIPS_VULKAN_ENABLED)
endif()

# ---- GPU tests ----
option(ELIPS_GPU_TESTS "Build GPU engine tests" OFF)
if(ELIPS_GPU_TESTS)
    add_subdirectory(tests)
endif()
