# Standalone smoke test: proves an installed ELIPS resolves via find_package and
# links through the exported elips::elips_core target. Built out-of-tree against
# an install prefix, NOT part of the main build. Usage:
#   cmake --install <build> --prefix <prefix>
#   cmake -S cmake/consumer_smoke -B <bld> -DCMAKE_PREFIX_PATH=<prefix>
#   cmake --build <bld> && <bld>/elips_consumer_smoke
cmake_minimum_required(VERSION 3.24)
project(elips_consumer_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(elips REQUIRED)

add_executable(elips_consumer_smoke main.cpp)
target_link_libraries(elips_consumer_smoke PRIVATE elips::elips_core)
