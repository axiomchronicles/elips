cmake_minimum_required(VERSION 3.24)
project(elips LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Static libraries are linked into the Python extension module, so build
# library objects as PIC by default across supported toolchains.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Enable Objective-C++ for Metal backend on Apple platforms
if(APPLE)
    enable_language(OBJCXX)
endif()

# ------------------------------- GPU engine -------------------------------

option(ELIPS_GPU_METAL "Enable Apple Metal GPU backend" ON)
option(ELIPS_GPU_CUDA "Enable NVIDIA CUDA GPU backend" OFF)
option(ELIPS_GPU_HIP "Enable AMD ROCm/HIP GPU backend" OFF)
option(ELIPS_GPU_SYCL "Enable Intel oneAPI/SYCL GPU backend" OFF)
option(ELIPS_GPU_VULKAN "Enable Vulkan compute GPU backend" OFF)

set(ELIPS_GPU_ENABLED OFF)
if(ELIPS_GPU_CUDA OR ELIPS_GPU_HIP OR ELIPS_GPU_METAL OR ELIPS_GPU_SYCL OR ELIPS_GPU_VULKAN)
    set(ELIPS_GPU_ENABLED ON)
endif()

if(ELIPS_GPU_ENABLED)
    add_subdirectory(src/gpu_engine)
endif()

# ----------------------------- core library ------------------------------
add_library(elips_core
    src/LocalTextEmbedder.cpp
    src/RecordID.cpp
    src/Metrics.cpp
    src/ExactIndex.cpp
    src/HierarchicalGraphIndex.cpp
    src/IndexFactory.cpp
    src/LockManager.cpp
    src/WAL.cpp
    src/Filter.cpp
    src/MetadataIndex.cpp
    src/EQLLexer.cpp
    src/EQLParser.cpp
    src/QueryExecutor.cpp
    src/Database.cpp
)
target_include_directories(elips_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(elips_core PRIVATE -Wall -Wextra -Wpedantic)

if(ELIPS_GPU_ENABLED)
    target_link_libraries(elips_core PUBLIC elips_gpu)
    target_compile_definitions(elips_core PUBLIC ELIPS_GPU_ENABLED)
endif()

# -------------------------------- elips-cli ------------------------------
option(ELIPS_BUILD_CLI "Build the elips command-line tool" ON)
if(ELIPS_BUILD_CLI)
    add_executable(elips_cli cli/src/main.cpp)
    set_target_properties(elips_cli PROPERTIES OUTPUT_NAME elips)
    target_link_libraries(elips_cli PRIVATE elips_core)
    target_compile_options(elips_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ------------------------------- benchmarks ------------------------------
option(ELIPS_BUILD_BENCH "Build the benchmark suite" ON)
if(ELIPS_BUILD_BENCH)
    add_executable(elips_bench benchmarks/BenchMain.cpp)
    target_link_libraries(elips_bench PRIVATE elips_core)
    target_compile_options(elips_bench PRIVATE -Wall -Wextra -Wpedantic)

    if(ELIPS_GPU_ENABLED)
        add_executable(elips_gpu_bench benchmarks/gpu/BenchGpuSearch.cpp)
        target_link_libraries(elips_gpu_bench PRIVATE elips_core)
        target_compile_definitions(elips_gpu_bench PRIVATE ELIPS_GPU_ENABLED)
        target_compile_options(elips_gpu_bench PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

# -------------------------------- tests ----------------------------------
option(ELIPS_BUILD_TESTS "Build the ELIPS test suite" ON)

if(ELIPS_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(elips_tests
        tests/unit/metrics_test.cpp
        tests/unit/exact_index_test.cpp
        tests/unit/record_id_test.cpp
        tests/unit/hnsw_recall_test.cpp
        tests/unit/eql_test.cpp
        tests/unit/vector_test.cpp
        tests/unit/config_test.cpp
        tests/unit/metadata_index_test.cpp
        tests/unit/exact_edge_cases_test.cpp
        tests/unit/hnsw_edge_cases_test.cpp
        tests/unit/filter_edge_test.cpp
        tests/unit/scan_test.cpp
        tests/unit/durability_test.cpp
        tests/unit/eql_edge_test.cpp
        tests/integration/roundtrip_test.cpp
        tests/integration/local_text_embedder_test.cpp
        tests/integration/transaction_filter_test.cpp
        tests/recovery/wal_recovery_test.cpp
        tests/concurrency/single_writer_test.cpp
        tests/concurrency/multi_reader_test.cpp
    )
    target_link_libraries(elips_tests PRIVATE elips_core GTest::gtest_main)

if(ELIPS_GPU_ENABLED)
    target_sources(elips_tests PRIVATE
        tests/unit/gpu/GpuDeviceManagerTest.cpp
        tests/unit/gpu/GpuBruteForceIndexTest.cpp
        tests/unit/gpu/GpuAdvancedIndexTest.cpp
        tests/unit/gpu/GpuMemoryManagerTest.cpp
        tests/unit/gpu/DynamicBatcherTest.cpp
        tests/integration/gpu/MetalBackendTest.cpp
        tests/parity/CpuGpuRecallParityTest.cpp
    )
    target_compile_definitions(elips_tests PRIVATE ELIPS_GPU_ENABLED)
    if(APPLE AND ELIPS_GPU_METAL)
        target_link_libraries(elips_tests PRIVATE elips_gpu_metal)
        find_library(METAL_FW Metal)
        find_library(MPS_FW MetalPerformanceShaders)
        find_library(FOUNDATION_FW Foundation)
        target_link_libraries(elips_tests PRIVATE ${METAL_FW} ${MPS_FW} ${FOUNDATION_FW})
    endif()

    add_executable(elips_gpu_lifecycle_probe
        tests/integration/gpu/DatabaseGpuLifecycleProbe.cpp
    )
    target_link_libraries(elips_gpu_lifecycle_probe PRIVATE elips_core)
    if(APPLE AND ELIPS_GPU_METAL)
        target_link_libraries(elips_gpu_lifecycle_probe PRIVATE elips_gpu_metal)
        target_link_libraries(elips_gpu_lifecycle_probe PRIVATE ${METAL_FW} ${MPS_FW} ${FOUNDATION_FW})
    endif()
    add_test(NAME GpuDatabaseLifecycleProbe COMMAND elips_gpu_lifecycle_probe)
endif()

    include(GoogleTest)
    gtest_discover_tests(elips_tests)
endif()

# ----------------------------- python bindings ---------------------------
option(ELIPS_BUILD_PYTHON "Build the Python (PyBind11) bindings" OFF)

if(ELIPS_BUILD_PYTHON)
    set(
        ELIPS_PYBIND11_SOURCE_DIR
        ""
        CACHE PATH
        "Optional local pybind11 source tree used instead of FetchContent"
    )
    if(ELIPS_PYBIND11_SOURCE_DIR)
        add_subdirectory(
            "${ELIPS_PYBIND11_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}/_deps/pybind11-local"
        )
    else()
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(elips_pymodule bindings/python/elips_python.cpp)
    target_link_libraries(elips_pymodule PRIVATE elips_core)
    if(ELIPS_GPU_ENABLED)
        target_compile_definitions(elips_pymodule PRIVATE ELIPS_GPU_ENABLED)
    endif()

    set(
        ELIPS_PYTHON_OUTPUT_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/elips"
        CACHE PATH
        "Directory where the Python extension module is written"
    )
    set_target_properties(elips_pymodule PROPERTIES
        OUTPUT_NAME _core
        LIBRARY_OUTPUT_DIRECTORY "${ELIPS_PYTHON_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${ELIPS_PYTHON_OUTPUT_DIR}"
    )
endif()
