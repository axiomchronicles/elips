# Metal backend for Apple GPUs (macOS / iOS)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

add_library(elips_gpu_metal STATIC
    MetalBackend.mm
)

target_include_directories(elips_gpu_metal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
)

target_compile_definitions(elips_gpu_metal PRIVATE ELIPS_METAL_ENABLED)

set_source_files_properties(MetalBackend.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc -O3 -ffast-math"
)
target_link_libraries(elips_gpu_metal PUBLIC
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

target_compile_options(elips_gpu_metal PRIVATE -Wall -Wextra -Wpedantic)

# ---- Precompile Metal shaders: .metal -> .air -> .metallib -----------------
# Adapted from FAISS faiss/gpu_metal/CMakeLists.txt (MIT, (c) Meta Platforms).
find_program(XCRUN xcrun REQUIRED)
set(ELIPS_METAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/elips_distance_kernels.metal)
set(ELIPS_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/elips_distance_kernels.air)
set(ELIPS_METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/elips_kernels.metallib)

add_custom_command(
    OUTPUT ${ELIPS_METAL_AIR}
    COMMAND ${XCRUN} -sdk macosx metal -c ${ELIPS_METAL_SOURCE} -o ${ELIPS_METAL_AIR}
    DEPENDS ${ELIPS_METAL_SOURCE}
    COMMENT "Compiling elips_distance_kernels.metal -> .air"
)
add_custom_command(
    OUTPUT ${ELIPS_METAL_LIB}
    COMMAND ${XCRUN} -sdk macosx metallib ${ELIPS_METAL_AIR} -o ${ELIPS_METAL_LIB}
    DEPENDS ${ELIPS_METAL_AIR}
    COMMENT "Linking elips_distance_kernels.air -> elips_kernels.metallib"
)
add_custom_target(elips_metal_shaders DEPENDS ${ELIPS_METAL_LIB})
add_dependencies(elips_gpu_metal elips_metal_shaders)

# Compiled-in path so dev builds (not installed) locate the metallib.
target_compile_definitions(elips_gpu_metal PRIVATE
    ELIPS_METALLIB_BUILD_PATH="${ELIPS_METAL_LIB}"
)

# Install the metallib beside the library for installed consumers (dladdr path).
include(GNUInstallDirs)
install(FILES ${ELIPS_METAL_LIB} DESTINATION ${CMAKE_INSTALL_LIBDIR})
