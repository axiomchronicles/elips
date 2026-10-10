# ELIPS minimal C API (opt-in via -DELIPS_ENABLE_C_API=ON).
add_library(elips_c STATIC elips_c_api.cpp)

# Public C header lives under include/elips/c_api; link elips_core for the impl.
target_include_directories(elips_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(elips_c PUBLIC elips_core)

if(MSVC)
    target_compile_options(elips_c PRIVATE /W4)
else()
    target_compile_options(elips_c PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Self-check: round-trips the id encoding and exercises create/add/search/free.
# Standalone (not a ctest) so it never perturbs the main suite count; run by
# hand during verification: build/src/c_api/elips_c_demo. Built whenever the C
# API is on (BUILD_TESTING is not yet defined at this point in configure).
add_executable(elips_c_demo demo.cpp)
target_link_libraries(elips_c_demo PRIVATE elips_c)
