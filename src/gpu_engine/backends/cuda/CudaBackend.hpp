#ifndef ELIPS_GPU_ENGINE_CUDA_BACKEND_HPP
#define ELIPS_GPU_ENGINE_CUDA_BACKEND_HPP

#ifdef ELIPS_CUDA_ENABLED

#include "elips/gpu_engine/GpuPort.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace elips::gpu::cuda {

class CudaBackend final : public GpuPort {
public:
    CudaBackend();
    ~CudaBackend() override;

    [[nodiscard]] std::expected<void, GpuError> initialize(const GpuConfig& config) override;
    void shutdown() noexcept override;

    [[nodiscard]] GpuDeviceInfo device_info() const noexcept override;
    [[nodiscard]] bool is_available() const noexcept override;

    [[nodiscard]] std::expected<GpuBuffer, GpuError> allocate_device(size_t bytes) override;
    void free_device(GpuBuffer&& buf) noexcept override;

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

    [[nodiscard]] std::expected<void, GpuError>
    gemm_fp32(std::span<const float> a, std::span<const float> b,
              std::span<float> c, size_t m, size_t n, size_t k_inner,
              bool trans_a, bool trans_b);

    void synchronize() override;
    [[nodiscard]] bool is_idle() const noexcept override;

    [[nodiscard]] cudaStream_t default_stream() const noexcept { return stream_; }

private:
    int device_index_{-1};
    cudaStream_t stream_{};
    bool initialized_{false};
    GpuDeviceInfo info_;
};

} // namespace elips::gpu::cuda

#endif // ELIPS_CUDA_ENABLED

#endif