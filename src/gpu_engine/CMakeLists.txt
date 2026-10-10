# ============================================================================
# ELIPS GPU Engine — CMake Build Module
# ============================================================================

add_library(elips_gpu STATIC
    GpuDeviceManager.cpp
    BackendSelector.cpp
    GpuIndexTransferManager.cpp
    GpuMemoryManager.cpp
    GpuMemoryPool.cpp
    QueryBatcher.cpp
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
    BruteForceKnn.cpp
)

target_include_directories(elips_gpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/hip
)

target_link_libraries(elips_gpu PUBLIC elips_core)
target_compile_options(elips_gpu PRIVATE -Wall -Wextra -Wpedantic)

# Backend flags are declared in the top-level CMakeLists.txt (single source of
# truth); this file only consumes them.

# ---- CUDA backend ----
if(ELIPS_ENABLE_CUDA)
    add_subdirectory(backends/cuda)
    target_link_libraries(elips_gpu PRIVATE elips_gpu_cuda)
    target_compile_definitions(elips_gpu PRIVATE ELIPS_CUDA_ENABLED)
endif()

# ---- HIP backend ----
if(ELIPS_ENABLE_ROCM)
    add_subdirectory(backends/hip)
    target_link_libraries(elips_gpu PRIVATE elips_gpu_hip)
    target_compile_definitions(elips_gpu PRIVATE ELIPS_HIP_ENABLED)
endif()

# ---- Metal backend ----
if(APPLE AND ELIPS_ENABLE_METAL)
    add_subdirectory(backends/metal)
    target_link_libraries(elips_gpu PUBLIC elips_gpu_metal)
    target_compile_definitions(elips_gpu PUBLIC ELIPS_METAL_ENABLED)
endif()

# ---- GPU tests ----
option(ELIPS_GPU_TESTS "Build GPU engine tests" OFF)
if(ELIPS_GPU_TESTS)
    add_subdirectory(tests)
endif()
