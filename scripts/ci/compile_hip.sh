#!/usr/bin/env bash
# ELIPS Phase 7.6 — HIP/ROCm backend container compile-check.
#
# EXPERIMENTAL — NOT HARDWARE VALIDATED. There is no AMD GPU in CI, so this is
# a COMPILE-ONLY proof: it runs a ROCm devel image and uses hipcc to compile
# the HIP translation units (amdgpu platform) against include/. It does NOT
# run anything on a GPU.
#
# On an arm64 host the amd64 ROCm image runs under emulation (slow) — allow a
# long timeout. On real x86_64 CI it compiles natively in seconds.
#
# Usage:
#   scripts/ci/compile_hip.sh                 # default image + arch
#   ROCM_IMAGE=rocm/dev-ubuntu-22.04:latest scripts/ci/compile_hip.sh
#   HIP_ARCH=gfx90a scripts/ci/compile_hip.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROCM_IMAGE="${ROCM_IMAGE:-rocm/dev-ubuntu-24.04:latest}"
HIP_ARCH="${HIP_ARCH:-gfx908}"   # MI100; any valid gfx target compiles w/o a GPU
PLATFORM="${PLATFORM:-linux/amd64}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found; cannot run the ROCm compile-check." >&2
    exit 2
fi

echo "== ELIPS HIP compile-check =="
echo "image:    ${ROCM_IMAGE}"
echo "platform: ${PLATFORM} (emulated + slow if host arch differs)"
echo "arch:     ${HIP_ARCH}"
echo "repo:     ${REPO_ROOT}"
echo

# Compile both .hip translation units to objects inside the container. hipcc is
# clang-based; -std=c++23 is required (the backend uses std::expected). hipBLAS
# headers live under the ROCm include prefix already on hipcc's search path.
docker run --rm --platform "${PLATFORM}" \
    -v "${REPO_ROOT}:/src:ro" \
    -w /src \
    "${ROCM_IMAGE}" \
    bash -c '
        set -euo pipefail
        export PATH=/opt/rocm/bin:$PATH
        echo "hipcc: $(command -v hipcc)"
        hipcc --version | head -n 3 || true
        mkdir -p /tmp/hipobj
        COMMON="-c -std=c++23 -O2 -DELIPS_HIP_ENABLED \
            --offload-arch='"${HIP_ARCH}"' \
            -I/src/include \
            -I/src/src/gpu_engine \
            -I/src/src/gpu_engine/backends/hip"
        set -x
        hipcc $COMMON /src/src/gpu_engine/backends/hip/kernels/cosine_fp32.hip \
            -o /tmp/hipobj/cosine_fp32.o
        hipcc $COMMON /src/src/gpu_engine/backends/hip/HipBackend.hip \
            -o /tmp/hipobj/HipBackend.o
        set +x
        echo "OK: HIP translation units compiled:"
        ls -l /tmp/hipobj
    '

echo
echo "== HIP compile-check PASSED =="
