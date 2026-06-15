#ifndef ELIPS_GPU_ENGINE_VULKAN_BACKEND_HPP
#define ELIPS_GPU_ENGINE_VULKAN_BACKEND_HPP

#ifdef ELIPS_VULKAN_ENABLED

#include "elips/gpu_engine/GpuPort.hpp"
#include <memory>
#include <vector>

namespace elips::gpu::vulkan {

class VulkanBackend final : public GpuPort {
public:
    VulkanBackend();
    ~VulkanBackend() override;

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

    void synchronize() override;
    [[nodiscard]] bool is_idle() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace elips::gpu::vulkan

#endif // ELIPS_VULKAN_ENABLED

#endif