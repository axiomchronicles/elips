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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/hip
)

# elips_gpu uses elips_core's headers (set above) and references a few of its
# symbols (Metrics, RecordID). It deliberately does NOT link elips_core: every
# final binary links elips_core directly, and elips_core PUBLIC-links elips_gpu,
# so a back-link here forms a static-lib cycle that makes ld64 (Xcode 15+) emit
# "ignoring duplicate libraries: libelips_core.a, libelips_gpu.a". ld64 resolves
# symbols order-independently, so dropping the back-link is safe on Apple.
# ponytail: if a GNU-ld GPU build (Linux CUDA/HIP) later reports undefined
# elips_core symbols from elips_gpu, wrap the pair in
# $<LINK_GROUP:RESCAN,elips_core,elips_gpu> on consumers rather than restoring
# the cycle.
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
# The GPU engine tests live in the top-level suite (built under ELIPS_GPU_ENABLED,
# see root CMakeLists.txt). There is no src/gpu_engine/tests/ subdir, so guard
# the opt-in to keep `-DELIPS_GPU_TESTS=ON` from breaking configure.
option(ELIPS_GPU_TESTS "Build GPU engine tests (now part of the top-level suite)" OFF)
if(ELIPS_GPU_TESTS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
        add_subdirectory(tests)
    else()
        message(WARNING "ELIPS_GPU_TESTS=ON but src/gpu_engine/tests/ is absent; "
                        "GPU tests are in the top-level suite. Ignoring.")
    endif()
endif()
