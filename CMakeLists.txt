if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum macOS deployment target")
endif()

cmake_minimum_required(VERSION 3.24)
project(elips LANGUAGES CXX VERSION 1.1.0)
# ^ project(VERSION) is the SINGLE SOURCE OF TRUTH for the ELIPS version.
# The Python extension receives it as the ELIPS_VERSION compile definition
# (see pybind section) and bindings/python/setup.py scrapes it from this line,
# so the version lives in exactly one place.

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Static libraries are linked into the Python extension module, so build
# library objects as PIC by default across supported toolchains.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# ------------------------------- sanitizers -------------------------------
# -DELIPS_SANITIZE=thread    validates the in-process locking (F6)
# -DELIPS_SANITIZE=address   validates memory safety (ASan + UBSan)
set(ELIPS_SANITIZE "" CACHE STRING "Sanitizer to build with: thread|address|none")
set_property(CACHE ELIPS_SANITIZE PROPERTY STRINGS "" none thread address)
if(ELIPS_SANITIZE STREQUAL "thread")
    add_compile_options(-fsanitize=thread -g -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
elseif(ELIPS_SANITIZE STREQUAL "address")
    add_compile_options(-fsanitize=address,undefined -g -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# Enable Objective-C++ for Metal backend on Apple platforms
if(APPLE)
    enable_language(OBJCXX)
endif()

# --------------------------- SIMD optimization ---------------------------
# ELIPS CPU distance kernels are hand-written NEON (arm64) / scalar fallbacks.
# ELIPS_OPT_LEVEL picks the baseline ISA the core library is compiled for.
# Unlike FAISS (which builds one static lib per tier for runtime dispatch), we
# build a single elips_core at the selected tier -- simpler, and enough until a
# portable multi-tier wheel is actually needed.
# ponytail: single-tier build now; add FAISS-style per-tier libs / runtime
# dispatch ("dd") only when one wheel must serve many x86 CPUs.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(_elips_opt_default "neon")
else()
    set(_elips_opt_default "generic")
endif()
set(ELIPS_OPT_LEVEL "${_elips_opt_default}" CACHE STRING
    "SIMD baseline for elips_core: generic|neon|avx2|avx512")
set_property(CACHE ELIPS_OPT_LEVEL PROPERTY STRINGS generic neon avx2 avx512)

# Resolved into ELIPS_OPT_FLAGS, applied to elips_core once it is defined.
set(ELIPS_OPT_FLAGS "")
if(ELIPS_OPT_LEVEL STREQUAL "generic")
    # Portable baseline: no extra ISA flags. NEON is still mandatory on arm64.
elseif(ELIPS_OPT_LEVEL STREQUAL "neon")
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
        message(WARNING "ELIPS_OPT_LEVEL=neon on a non-arm64 target; NEON is "
                        "the arm64 baseline. Treating as generic.")
    endif()
    # NEON is baseline-mandatory on arm64 (AArch64 ABI) -- no flag needed.
elseif(ELIPS_OPT_LEVEL STREQUAL "avx2")
    if(MSVC)
        set(ELIPS_OPT_FLAGS /arch:AVX2)
    else()
        set(ELIPS_OPT_FLAGS -mavx2 -mfma -mf16c)
    endif()
elseif(ELIPS_OPT_LEVEL STREQUAL "avx512")
    if(MSVC)
        set(ELIPS_OPT_FLAGS /arch:AVX512)
    else()
        set(ELIPS_OPT_FLAGS -mavx2 -mfma -mf16c
            -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)
    endif()
else()
    message(FATAL_ERROR "Unknown ELIPS_OPT_LEVEL '${ELIPS_OPT_LEVEL}' "
                        "(expected generic|neon|avx2|avx512)")
endif()
message(STATUS "ELIPS_OPT_LEVEL=${ELIPS_OPT_LEVEL} flags='${ELIPS_OPT_FLAGS}'")

# ------------------------------- BLAS (future) ---------------------------
# ELIPS distance is hand/NEON, NOT BLAS -- there is no gemm consumer today, so
# this stays OFF and is a deliberate no-op. It exists only to reserve the knob
# for a future batched gemm distance path (then: find_package(BLAS) + link).
# ponytail: no-op until a gemm distance kernel actually exists.
option(ELIPS_ENABLE_BLAS "Reserve BLAS linkage for a future gemm path (no-op today)" OFF)
if(ELIPS_ENABLE_BLAS)
    message(WARNING "ELIPS_ENABLE_BLAS=ON has no effect: ELIPS has no BLAS "
                    "consumer yet. Ignoring.")
endif()

# ------------------------------- GPU engine -------------------------------
# Backend flags live here only; src/gpu_engine/CMakeLists.txt consumes them.
# Declaring option() in both places is a trap: whichever runs first wins the
# cache, so changing a default in one file silently has no effect.
#
# Metal defaults ON only on Apple. Defaulting it ON everywhere made a plain
# `cmake -S . -B build` on Linux set ELIPS_GPU_ENABLED, build the whole
# elips_gpu library, and expose gpu_info()/gpu_stats() with no backend behind
# them.
# Back-compat: honor the old option names (deprecated). Each alias must run
# before the matching option() so the forced cache value wins.
if(DEFINED ELIPS_GPU_METAL AND NOT DEFINED ELIPS_ENABLE_METAL)
    set(ELIPS_ENABLE_METAL "${ELIPS_GPU_METAL}" CACHE BOOL "" FORCE)
    message(DEPRECATION "ELIPS_GPU_METAL is deprecated; use ELIPS_ENABLE_METAL")
endif()
if(DEFINED ELIPS_GPU_CUDA AND NOT DEFINED ELIPS_ENABLE_CUDA)
    set(ELIPS_ENABLE_CUDA "${ELIPS_GPU_CUDA}" CACHE BOOL "" FORCE)
    message(DEPRECATION "ELIPS_GPU_CUDA is deprecated; use ELIPS_ENABLE_CUDA")
endif()
if(DEFINED ELIPS_GPU_HIP AND NOT DEFINED ELIPS_ENABLE_ROCM)
    set(ELIPS_ENABLE_ROCM "${ELIPS_GPU_HIP}" CACHE BOOL "" FORCE)
    message(DEPRECATION "ELIPS_GPU_HIP is deprecated; use ELIPS_ENABLE_ROCM")
endif()

option(ELIPS_ENABLE_METAL "Enable Apple Metal GPU backend" ${APPLE})
option(ELIPS_ENABLE_CUDA "Enable NVIDIA CUDA GPU backend" OFF)
option(ELIPS_ENABLE_ROCM "Enable AMD ROCm/HIP GPU backend" OFF)

if(ELIPS_ENABLE_METAL AND NOT APPLE)
    message(WARNING
        "ELIPS_ENABLE_METAL is ON but this is not an Apple platform; "
        "no Metal backend will be compiled.")
endif()

set(ELIPS_GPU_ENABLED OFF)
if(ELIPS_ENABLE_CUDA OR ELIPS_ENABLE_ROCM OR (ELIPS_ENABLE_METAL AND APPLE))
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
    src/IndexFlat.cpp
    src/IndexHNSW.cpp
    src/IndexIVFFlat.cpp
    src/IndexIVFPQ.cpp
    src/IndexIVFScalarQuantizer.cpp
    src/IndexFactory.cpp
    src/quant_engine/KMeans.cpp
    src/quant_engine/Quantizer.cpp
    src/quant_engine/ProductQuantizer.cpp
    src/quant_engine/ScalarQuantizer.cpp
    src/LockManager.cpp
    src/WAL.cpp
    src/Filter.cpp
    src/MetadataFilterIndex.cpp
    src/EQLLexer.cpp
    src/EQLParser.cpp
    src/QueryExecutor.cpp
    src/Database.cpp
)
target_include_directories(elips_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# SIMD baseline selected by ELIPS_OPT_LEVEL (resolved above).
if(ELIPS_OPT_FLAGS)
    target_compile_options(elips_core PRIVATE ${ELIPS_OPT_FLAGS})
endif()
if(WIN32)
    target_compile_definitions(elips_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
if(MSVC)
    target_compile_options(elips_core PRIVATE /W4)
else()
    target_compile_options(elips_core PRIVATE -Wall -Wextra -Wpedantic)
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(elips_core PRIVATE -Wa,-mbig-obj)
    endif()
endif()

if(ELIPS_GPU_ENABLED)
    target_link_libraries(elips_core PUBLIC elips_gpu)
    target_compile_definitions(elips_core PUBLIC ELIPS_GPU_ENABLED)
endif()

# ------------------------------- C API -----------------------------------
# Thin extern "C" shim over the index_factory/add/search lifecycle. Off by
# default (mirrors FAISS_ENABLE_C_API); its own subdir keeps the surface small.
option(ELIPS_ENABLE_C_API "Build the minimal C API wrapper (elips_c)" OFF)
if(ELIPS_ENABLE_C_API)
    add_subdirectory(src/c_api)
endif()

# -------------------------------- elips-cli ------------------------------
option(ELIPS_BUILD_CLI "Build the elips command-line tool" ON)
if(ELIPS_BUILD_CLI)
    add_executable(elips_cli cli/src/main.cpp)
    set_target_properties(elips_cli PROPERTIES OUTPUT_NAME elips)
    target_link_libraries(elips_cli PRIVATE elips_core)
    if(MSVC)
        target_compile_options(elips_cli PRIVATE /W4)
    else()
        target_compile_options(elips_cli PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

# ------------------------------- benchmarks ------------------------------
option(ELIPS_BUILD_BENCH "Build the benchmark suite" ON)
if(ELIPS_BUILD_BENCH)
    add_executable(elips_bench benchmarks/BenchMain.cpp)
    target_link_libraries(elips_bench PRIVATE elips_core)
    if(MSVC)
        target_compile_options(elips_bench PRIVATE /W4)
    else()
        target_compile_options(elips_bench PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(ELIPS_GPU_ENABLED)
        add_executable(elips_gpu_bench benchmarks/gpu/BenchGpuSearch.cpp)
        target_link_libraries(elips_gpu_bench PRIVATE elips_core)
        target_compile_definitions(elips_gpu_bench PRIVATE ELIPS_GPU_ENABLED)
        if(MSVC)
            target_compile_options(elips_gpu_bench PRIVATE /W4)
        else()
            target_compile_options(elips_gpu_bench PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endif()
endif()

# -------------------------------- tests ----------------------------------
if(DEFINED ELIPS_BUILD_TESTS AND NOT DEFINED BUILD_TESTING)
    set(BUILD_TESTING "${ELIPS_BUILD_TESTS}" CACHE BOOL "" FORCE)
    message(DEPRECATION "ELIPS_BUILD_TESTS is deprecated; use BUILD_TESTING")
endif()
option(BUILD_TESTING "Build the ELIPS test suite" ON)

if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    # Don't let GoogleTest's own install() rules leak gtest/gmock into the ELIPS
    # package prefix.
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(elips_tests
        tests/unit/metrics_test.cpp
        tests/unit/index_flat_test.cpp
        tests/unit/record_id_test.cpp
        tests/unit/hnsw_recall_test.cpp
        tests/unit/eql_test.cpp
        tests/unit/vector_test.cpp
        tests/unit/config_test.cpp
        tests/unit/metadata_filter_index_test.cpp
        tests/unit/index_flat_edge_cases_test.cpp
        tests/unit/index_ivfflat_test.cpp
        tests/unit/index_ivfpq_test.cpp
        tests/unit/index_ivfsq_test.cpp
        tests/unit/index_factory_test.cpp
        tests/unit/typed_errors_test.cpp
        tests/unit/range_search_test.cpp
        tests/unit/hnsw_edge_cases_test.cpp
        tests/unit/filter_edge_test.cpp
        tests/unit/scan_test.cpp
        tests/unit/durability_test.cpp
        tests/unit/m1_durability_atomicity_test.cpp
        tests/unit/m3_index_health_test.cpp
        tests/unit/eql_edge_test.cpp
        tests/unit/quantizer_test.cpp
        tests/unit/quantized_index_test.cpp
        tests/unit/m5_quantization_persistence_test.cpp
        tests/integration/roundtrip_test.cpp
        tests/integration/local_text_embedder_test.cpp
        tests/integration/transaction_filter_test.cpp
        tests/recovery/wal_recovery_test.cpp
        tests/concurrency/single_writer_test.cpp
        tests/concurrency/multi_reader_test.cpp
        tests/concurrency/m2_thread_safety_test.cpp
        tests/fuzz/parser_robustness_test.cpp
    )
    target_link_libraries(elips_tests PRIVATE elips_core GTest::gtest_main)

if(ELIPS_GPU_ENABLED)
    target_sources(elips_tests PRIVATE
        tests/unit/gpu/GpuDeviceManagerTest.cpp
        tests/unit/gpu/GpuBruteForceIndexTest.cpp
        tests/unit/gpu/GpuAdvancedIndexTest.cpp
        tests/unit/gpu/GpuMemoryManagerTest.cpp
        tests/unit/gpu/m4_memory_manager_test.cpp
        tests/unit/gpu/QueryBatcherTest.cpp
        tests/integration/gpu/MetalBackendTest.cpp
        tests/parity/CpuGpuRecallParityTest.cpp
    )
    target_compile_definitions(elips_tests PRIVATE ELIPS_GPU_ENABLED)
    if(APPLE AND ELIPS_ENABLE_METAL)
        target_sources(elips_tests PRIVATE
            tests/integration/gpu/MetalRealKernelParityTest.cpp
            tests/integration/gpu/MetalBruteForceKnnParityTest.cpp
            tests/integration/gpu/MetalIvfParityTest.cpp)
        # elips_gpu_metal arrives transitively (elips_core -> elips_gpu ->
        # elips_gpu_metal, all PUBLIC); linking it directly too made ld64 warn
        # "ignoring duplicate libraries". Only MPS (not re-exported by the
        # backend) needs an explicit link here.
        find_library(METAL_FW Metal)
        find_library(MPS_FW MetalPerformanceShaders)
        find_library(FOUNDATION_FW Foundation)
        target_link_libraries(elips_tests PRIVATE ${METAL_FW} ${MPS_FW} ${FOUNDATION_FW})
    endif()
    # CUDA parity test — compiled only with the CUDA backend; skips without a device.
    if(ELIPS_ENABLE_CUDA)
        target_sources(elips_tests PRIVATE tests/gpu/cuda/CudaBackendTest.cpp)
        target_compile_definitions(elips_tests PRIVATE ELIPS_CUDA_ENABLED)
        target_link_libraries(elips_tests PRIVATE elips_gpu_cuda)
    endif()

    # HIP/ROCm parity test — compiled only when the HIP backend is enabled,
    # skipped at runtime unless an AMD GPU is present (EXPERIMENTAL).
    if(ELIPS_ENABLE_ROCM)
        target_sources(elips_tests PRIVATE tests/gpu/hip/HipBackendTest.cpp)
        target_compile_definitions(elips_tests PRIVATE ELIPS_HIP_ENABLED)
        target_link_libraries(elips_tests PRIVATE elips_gpu_hip)
    endif()

    add_executable(elips_gpu_lifecycle_probe
        tests/integration/gpu/DatabaseGpuLifecycleProbe.cpp
    )
    target_link_libraries(elips_gpu_lifecycle_probe PRIVATE elips_core)
    if(APPLE AND ELIPS_ENABLE_METAL)
        # elips_gpu_metal is transitive via elips_core (see note above).
        target_link_libraries(elips_gpu_lifecycle_probe PRIVATE ${METAL_FW} ${MPS_FW} ${FOUNDATION_FW})
    endif()
    add_test(NAME GpuDatabaseLifecycleProbe COMMAND elips_gpu_lifecycle_probe)
endif()

    include(GoogleTest)
    gtest_discover_tests(elips_tests)
endif()

# ------------------------------- fuzzers ---------------------------------
# libFuzzer targets for the on-disk parsers. Clang only; the deterministic
# mutation tests in tests/fuzz/parser_robustness_test.cpp cover the same surface
# in the normal suite.
option(ELIPS_BUILD_FUZZERS "Build libFuzzer targets (requires Clang)" OFF)
if(ELIPS_BUILD_FUZZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "ELIPS_BUILD_FUZZERS requires Clang (libFuzzer).")
    endif()
    add_executable(elips_fuzz_wal tests/fuzz/fuzz_wal_replay.cpp)
    target_link_libraries(elips_fuzz_wal PRIVATE elips_core)
    target_compile_options(elips_fuzz_wal PRIVATE
        -fsanitize=fuzzer,address,undefined -g)
    target_link_options(elips_fuzz_wal PRIVATE
        -fsanitize=fuzzer,address,undefined)
endif()

# ----------------------------- python bindings ---------------------------
if(DEFINED ELIPS_BUILD_PYTHON AND NOT DEFINED ELIPS_ENABLE_PYTHON)
    set(ELIPS_ENABLE_PYTHON "${ELIPS_BUILD_PYTHON}" CACHE BOOL "" FORCE)
    message(DEPRECATION "ELIPS_BUILD_PYTHON is deprecated; use ELIPS_ENABLE_PYTHON")
endif()
option(ELIPS_ENABLE_PYTHON "Build the Python (PyBind11) bindings" OFF)

if(ELIPS_ENABLE_PYTHON)
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
    # Version flows from project(VERSION) -> extension -> elips.__version__,
    # so there is nothing to hand-sync.
    target_compile_definitions(elips_pymodule PRIVATE ELIPS_VERSION="${PROJECT_VERSION}")
    if(ELIPS_GPU_ENABLED)
        target_compile_definitions(elips_pymodule PRIVATE ELIPS_GPU_ENABLED)
    endif()
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(elips_pymodule PRIVATE -Wa,-mbig-obj)
        target_link_options(elips_pymodule PRIVATE -static -static-libgcc -static-libstdc++)
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
        ARCHIVE_OUTPUT_DIRECTORY "${ELIPS_PYTHON_OUTPUT_DIR}"
    )
    if(CMAKE_CONFIGURATION_TYPES)
        foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER ${CONFIG} CONFIG_UPPER)
            set_target_properties(elips_pymodule PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${ELIPS_PYTHON_OUTPUT_DIR}"
                RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${ELIPS_PYTHON_OUTPUT_DIR}"
                ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${ELIPS_PYTHON_OUTPUT_DIR}"
            )
        endforeach()
    endif()
    if(CMAKE_BUILD_TYPE)
        string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
        set_target_properties(elips_pymodule PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE_UPPER} "${ELIPS_PYTHON_OUTPUT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE_UPPER} "${ELIPS_PYTHON_OUTPUT_DIR}"
            ARCHIVE_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE_UPPER} "${ELIPS_PYTHON_OUTPUT_DIR}"
        )
    endif()
endif()

# ------------------------- install / export ------------------------------
# Mirrors FAISS's install convention (faiss/CMakeLists.txt:778-841): one export
# set, a generated config package + version file, so downstream projects do
#     find_package(elips REQUIRED)
#     target_link_libraries(app PRIVATE elips::elips_core)
# Every library that ends up on elips_core's PUBLIC interface must share the
# export set, otherwise install(EXPORT) errors. elips_core PUBLIC-links elips_gpu
# (GPU builds), which PUBLIC-links elips_gpu_metal (Apple) -- so those join too.
include(CMakePackageConfigHelpers)

set(ELIPS_INSTALL_TARGETS elips_core)
if(ELIPS_GPU_ENABLED)
    list(APPEND ELIPS_INSTALL_TARGETS elips_gpu)
endif()
if(APPLE AND ELIPS_ENABLE_METAL AND ELIPS_GPU_ENABLED)
    list(APPEND ELIPS_INSTALL_TARGETS elips_gpu_metal)
endif()
if(ELIPS_ENABLE_CUDA AND ELIPS_GPU_ENABLED)
    list(APPEND ELIPS_INSTALL_TARGETS elips_gpu_cuda)
endif()
if(ELIPS_ENABLE_ROCM AND ELIPS_GPU_ENABLED)
    list(APPEND ELIPS_INSTALL_TARGETS elips_gpu_hip)
endif()
if(TARGET elips_c)
    list(APPEND ELIPS_INSTALL_TARGETS elips_c)
endif()

install(TARGETS ${ELIPS_INSTALL_TARGETS}
    EXPORT elipsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/elips
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT elipsTargets
    FILE elipsTargets.cmake
    NAMESPACE elips::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/elips)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/elipsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/elipsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/elips)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/elipsConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/elipsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/elipsConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/elips)

