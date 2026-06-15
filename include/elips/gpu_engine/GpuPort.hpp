#ifndef ELIPS_GPU_ENGINE_GPU_PORT_HPP
#define ELIPS_GPU_ENGINE_GPU_PORT_HPP

#include <expected>
#include <span>
#include <string_view>

#include "elips/Config.hpp"
#include "elips/gpu_engine/GpuBuffer.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuDeviceInfo.hpp"

namespace elips::gpu {

enum class GpuError {
    DeviceNotFound,
    InsufficientMemory,
    KernelLaunchFailed,
    TransferFailed,
    IndexBuildFailed,
    UnsupportedMetric,
    InitializationFailed,
    BackendUnavailable,
};

[[nodiscard]] constexpr std::string_view to_string(GpuError e) noexcept {
    switch (e) {
        case GpuError::DeviceNotFound:       return "DeviceNotFound";
        case GpuError::InsufficientMemory:   return "InsufficientMemory";
        case GpuError::KernelLaunchFailed:   return "KernelLaunchFailed";
        case GpuError::TransferFailed:       return "TransferFailed";
        case GpuError::IndexBuildFailed:     return "IndexBuildFailed";
        case GpuError::UnsupportedMetric:    return "UnsupportedMetric";
        case GpuError::InitializationFailed: return "InitializationFailed";
        case GpuError::BackendUnavailable:   return "BackendUnavailable";
    }
    return "Unknown";
}

class GpuPort {
public:
    virtual ~GpuPort() = default;

    [[nodiscard]] virtual std::expected<void, GpuError>
    initialize(const GpuConfig& config) = 0;

    virtual void shutdown() noexcept = 0;

    [[nodiscard]] virtual GpuDeviceInfo device_info() const noexcept = 0;
    [[nodiscard]] virtual bool is_available() const noexcept = 0;

    [[nodiscard]] virtual std::expected<GpuBuffer, GpuError>
    allocate_device(size_t bytes) = 0;

    virtual void free_device(GpuBuffer&& buf) noexcept = 0;

    [[nodiscard]] virtual std::expected<void, GpuError>
    upload(const void* host_src, GpuBuffer& dst, size_t bytes) = 0;

    [[nodiscard]] virtual std::expected<void, GpuError>
    download(const GpuBuffer& src, void* host_dst, size_t bytes) = 0;

    [[nodiscard]] virtual std::expected<void, GpuError>
    compute_distances_batch(
        std::span<const float> queries,
        std::span<const float> database,
        std::span<float> distances_out,
        size_t nq,
        size_t nb,
        size_t dim,
        elips::Metric metric) = 0;

    [[nodiscard]] virtual std::expected<void, GpuError>
    top_k(
        std::span<const float> distances,
        std::span<uint32_t> indices_out,
        std::span<float> values_out,
        size_t nq,
        size_t nb,
        size_t k) = 0;

    virtual void synchronize() = 0;
    [[nodiscard]] virtual bool is_idle() const noexcept = 0;
};

} // namespace elips::gpu

#endif