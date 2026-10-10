#!/usr/bin/env bash
# ============================================================================
# ELIPS HIP/ROCm backend compile-check  —  EXPERIMENTAL / NOT HARDWARE-VALIDATED
# ----------------------------------------------------------------------------
# Configures + builds ONLY the HIP backend target (elips_gpu_hip: HipBackend.hip
# + kernels/*.hip) inside an AMD ROCm devel container, via the real project
# CMake wiring (-DELIPS_ENABLE_ROCM=ON -> find_package(hip)+hipblas, HIP
# language). No GPU is required: this proves the code compiles with hipcc +
# hipBLAS against the current GpuBackend interface, NOT that it runs. For an
# actual runtime check, run the gated tests/gpu/hip suite on an x86_64 host with
# an AMD GPU. Mirrors scripts/ci/compile_cuda.sh.
#
# On an arm64 host the amd64 ROCm image runs under emulation (slow) — allow a
# long timeout. On real x86_64 CI it compiles natively in seconds.
#
# Usage:  scripts/ci/compile_hip.sh
# Env overrides:
#   ROCM_IMAGE        container image (default rocm/dev-ubuntu-24.04:latest)
#   DOCKER_PLATFORM   platform (default linux/amd64; set empty for native)
#   HIP_ARCH          CMAKE_HIP_ARCHITECTURES (default gfx908; any valid gfx
#                     target compiles without a physical GPU)
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROCM_IMAGE="${ROCM_IMAGE:-rocm/dev-ubuntu-24.04:latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
HIP_ARCH="${HIP_ARCH:-gfx908}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found; cannot run the ROCm compile-check." >&2
    exit 2
fi

PLATFORM_ARG=()
if [[ -n "${DOCKER_PLATFORM}" ]]; then
    PLATFORM_ARG=(--platform "${DOCKER_PLATFORM}")
fi

echo ">> ELIPS HIP compile-check"
echo ">> image=${ROCM_IMAGE} platform=${DOCKER_PLATFORM:-native} arch=${HIP_ARCH}"
echo ">> repo=${REPO_ROOT}"

# The build runs entirely inside the container against the mounted, read-only
# source tree; artifacts land in a container-local /build (never the host repo).
docker run --rm "${PLATFORM_ARG[@]}" \
    -v "${REPO_ROOT}:/src:ro" \
    -e HIP_ARCH="${HIP_ARCH}" \
    "${ROCM_IMAGE}" \
    bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive
        export PATH=/opt/rocm/bin:$PATH
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends cmake ninja-build g++ >/dev/null
        echo ">> toolchain:"; hipcc --version | head -1; cmake --version | head -1
        cmake -S /src -B /build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DELIPS_ENABLE_ROCM=ON \
            -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
            -DCMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" \
            -DELIPS_BUILD_TESTS=OFF \
            -DELIPS_BUILD_CLI=OFF \
            -DELIPS_BUILD_BENCH=OFF \
            -DELIPS_BUILD_PYTHON=OFF
        echo ">> building target elips_gpu_hip"
        cmake --build /build --target elips_gpu_hip -j"$(nproc)"
        echo ">> HIP backend objects:"
        find /build -name "*.hip.o" -o -name "libelips_gpu_hip*" | sort
        echo ">> HIP compile-check PASSED"
    '
