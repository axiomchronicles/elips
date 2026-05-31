cmake_minimum_required(VERSION 3.24)
project(elips LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# ----------------------------- core library ------------------------------
add_library(elips_core
    src/RecordID.cpp
    src/Metrics.cpp
    src/ExactIndex.cpp
    src/HierarchicalGraphIndex.cpp
    src/IndexFactory.cpp
    src/LockManager.cpp
    src/WAL.cpp
    src/Filter.cpp
    src/EQLLexer.cpp
    src/EQLParser.cpp
    src/QueryExecutor.cpp
    src/Database.cpp
)
target_include_directories(elips_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(elips_core PRIVATE -Wall -Wextra -Wpedantic)

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
        tests/integration/roundtrip_test.cpp
        tests/integration/transaction_filter_test.cpp
        tests/recovery/wal_recovery_test.cpp
        tests/concurrency/single_writer_test.cpp
    )
    target_link_libraries(elips_tests PRIVATE elips_core GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(elips_tests)
endif()

# ----------------------------- python bindings ---------------------------
option(ELIPS_BUILD_PYTHON "Build the Python (PyBind11) bindings" OFF)

if(ELIPS_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(elips_pymodule bindings/python/elips_python.cpp)
    target_link_libraries(elips_pymodule PRIVATE elips_core)
    set_target_properties(elips_pymodule PROPERTIES
        OUTPUT_NAME _core
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/elips
    )
endif()
