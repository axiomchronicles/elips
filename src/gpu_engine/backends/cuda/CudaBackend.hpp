// ============================================================================
// ELIPS CUDA GPU backend  —  EXPERIMENTAL / NOT HARDWARE-VALIDATED
// ----------------------------------------------------------------------------
// Status: EXPERIMENTAL. This backend is compiled only when ELIPS_GPU_CUDA=ON
// (OFF by default) and must be BUILT FROM SOURCE with the NVIDIA CUDA Toolkit
// (nvcc). It has been COMPILE-CHECKED in an nvidia/cuda devel container but has
// NOT been validated on real NVIDIA hardware. device_info().name is suffixed
// with "[EXPERIMENTAL]" to make the maturity level visible at runtime.
//
// Device k-selection adapted from the FAISS GPU block/warp-select design
// (Meta Platforms, MIT License) — see kernels/elips_cuda_kernels.cu.
// ============================================================================
#ifndef ELIPS_GPU_ENGINE_CUDA_BACKEND_HPP
#define ELIPS_GPU_ENGINE_CUDA_BACKEND_HPP

#ifdef ELIPS_CUDA_ENABLED

#include "elips/gpu_engine/GpuBackend.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace elips::gpu::cuda {

// Maturity marker surfaced in device_info().name and headers/logs.
inline constexpr const char* kCudaMaturity = "EXPERIMENTAL";

class CudaBackend final : public GpuBackend {
public:
    CudaBackend();
    ~CudaBackend() override;

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

    // ivf_scan() intentionally not overridden: CUDA uses the GpuBackend default
    // (BackendUnavailable), so the IVF index falls back to its generic path.

    void synchronize() override;
    [[nodiscard]] bool is_idle() const noexcept override;

    [[nodiscard]] cudaStream_t default_stream() const noexcept { return stream_; }

private:
    int device_index_{-1};
    cudaStream_t stream_{};
    cublasHandle_t cublas_{};
    bool initialized_{false};
    GpuDeviceInfo info_;
};

} // namespace elips::gpu::cuda

#endif // ELIPS_CUDA_ENABLED

#endif
