#ifndef ELIPS_GPU_ENGINE_HIP_BACKEND_HPP
#define ELIPS_GPU_ENGINE_HIP_BACKEND_HPP

#ifdef ELIPS_HIP_ENABLED

// ELIPS HIP/ROCm GPU backend.
//
// EXPERIMENTAL — NOT HARDWARE VALIDATED. This backend is compile-checked only
// (no AMD GPU in CI). It mirrors the CUDA backend using the HIP portability
// APIs (hipMalloc / hipMemcpy / hipStream*) plus hipBLAS SGEMM for batched
// distance computation and a device k-selection kernel. Like FAISS's ROCm
// support (whose GPU code is hipified from the CUDA sources and ships
// build-from-source rather than as a prebuilt conda package), this must be
// built from source against a ROCm SDK and validated on real AMD hardware
// before any production use.
//
// This header is deliberately free of any HIP types so it can be included by
// the regular C++ compiler that builds GpuDeviceManager.cpp. All HIP state
// lives behind the pimpl, defined in HipBackend.hip (compiled by hipcc).

#include "elips/gpu_engine/GpuPort.hpp"
#include <memory>

namespace elips::gpu::hip {

class HipBackend final : public GpuPort {
public:
    HipBackend();
    ~HipBackend() override;

    [[nodiscard]] std::expected<void, GpuError> initialize(const GpuConfig& config) override;
    void shutdown() noexcept override;

    [[nodiscard]] GpuDeviceInfo device_info() const noexcept override;
    [[nodiscard]] bool is_available() const noexcept override;

    [[nodiscard]] std::expected<GpuBuffer, GpuError> allocate_device(size_t bytes) override;
    void free_device(GpuBuffer buf) noexcept override;

    [[nodiscard]] std::expected<void, GpuError>
    upload(const void* host_src, GpuBuffer& dst, size_t bytes) override;

    [[nodiscard]] std::expected<void, GpuError>
    download(const GpuBuffer& src, void* host_dst, size_t bytes) override;

    [[nodiscard]] std::expected<void, GpuError>
    compute_distances_batch(std::span<const float> queries, std::span<const float> database,
                            std::span<float> distances_out, size_t nq, size_t nb, size_t dim,
                            elips::Metric metric) override;

    [[nodiscard]] std::expected<void, GpuError>
    top_k(std::span<const float> distances, std::span<uint32_t> indices_out,
          std::span<float> values_out, size_t nq, size_t nb, size_t k) override;

    void synchronize() override;
    [[nodiscard]] bool is_idle() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace elips::gpu::hip

#endif // ELIPS_HIP_ENABLED

#endif
