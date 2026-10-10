#!/usr/bin/env bash
# ============================================================================
# ELIPS CUDA backend compile-check  —  EXPERIMENTAL / NOT HARDWARE-VALIDATED
# ----------------------------------------------------------------------------
# Compiles ONLY the CUDA backend translation units (CudaBackend.cu +
# kernels/*.cu) inside an NVIDIA CUDA devel container. No GPU is required: this
# proves the code compiles with nvcc + cuBLAS, not that it runs. For an actual
# runtime check, run the gated test suite (tests/gpu/cuda) on an x86_64 host
# with an NVIDIA GPU.
#
# Usage:  scripts/ci/compile_cuda.sh
# Env overrides:
#   CUDA_IMAGE   container image (default nvidia/cuda:12.6.2-devel-ubuntu24.04)
#   DOCKER_PLATFORM  platform (default linux/amd64; set empty for native)
#   CUDA_ARCHS   CMAKE_CUDA_ARCHITECTURES (default 70;80;90)
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.6.2-devel-ubuntu24.04}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
CUDA_ARCHS="${CUDA_ARCHS:-70;80;90}"

PLATFORM_ARG=()
if [[ -n "${DOCKER_PLATFORM}" ]]; then
    PLATFORM_ARG=(--platform "${DOCKER_PLATFORM}")
fi

echo ">> ELIPS CUDA compile-check"
echo ">> image=${CUDA_IMAGE} platform=${DOCKER_PLATFORM:-native} archs=${CUDA_ARCHS}"
echo ">> repo=${REPO_ROOT}"

# The build runs entirely inside the container against the mounted, read-only
# source tree; artifacts land in a container-local /build (never the host repo).
docker run --rm "${PLATFORM_ARG[@]}" \
    -v "${REPO_ROOT}:/src:ro" \
    -e CUDA_ARCHS="${CUDA_ARCHS}" \
    "${CUDA_IMAGE}" \
    bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends cmake ninja-build g++ >/dev/null
        echo ">> toolchain:"; nvcc --version | tail -1; cmake --version | head -1
        cmake -S /src -B /build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DELIPS_ENABLE_CUDA=ON \
            -DCMAKE_CUDA_COMPILER=nvcc \
            -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
            -DELIPS_BUILD_TESTS=OFF \
            -DELIPS_BUILD_CLI=OFF \
            -DELIPS_BUILD_BENCH=OFF \
            -DELIPS_BUILD_PYTHON=OFF
        echo ">> building target elips_gpu_cuda"
        cmake --build /build --target elips_gpu_cuda -j"$(nproc)"
        echo ">> CUDA backend objects:"
        find /build -name "*.cu.o" -o -name "libelips_gpu_cuda*" | sort
        echo ">> CUDA compile-check PASSED"
    '
