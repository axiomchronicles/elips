#ifndef ELIPS_GPU_ENGINE_GPU_BACKEND_HPP
#define ELIPS_GPU_ENGINE_GPU_BACKEND_HPP

#include <span>
#include <string_view>

#include "elips/Config.hpp"
#include "elips/domain/Expected.hpp"
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

class GpuBackend {
public:
    virtual ~GpuBackend() = default;

    [[nodiscard]] virtual std::expected<void, GpuError>
    initialize(const GpuConfig& config) = 0;

    virtual void shutdown() noexcept = 0;

    [[nodiscard]] virtual GpuDeviceInfo device_info() const noexcept = 0;
    [[nodiscard]] virtual bool is_available() const noexcept = 0;

    [[nodiscard]] virtual std::expected<GpuBuffer, GpuError>
    allocate_device(size_t bytes) = 0;

    // Consume buffers by value so ownership-transfer call sites actually clear
    // their moved-from state before any repeated cleanup path runs.
    virtual void free_device(GpuBuffer buf) noexcept = 0;

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

    // GPU IVF list-scan + top-k (two-pass: per-list scan then merge). The
    // candidate vectors are the host-gathered probed lists laid out
    // contiguously; list_offsets (size nlists+1) bound each list's rows. The
    // index written into indices_out is the GLOBAL candidate row (0..ncand), so
    // the caller maps it back to a record id. Defaulted to "unsupported" so a
    // backend without an IVF-scan kernel (CUDA/HIP today) is left
    // untouched and the IVF index falls back to its generic scan path.
    [[nodiscard]] virtual std::expected<void, GpuError>
    ivf_scan(
        std::span<const float> query,
        std::span<const float> candidate_vectors,
        std::span<const uint32_t> list_offsets,
        std::span<uint32_t> indices_out,
        std::span<float> values_out,
        size_t nlists,
        size_t ncand,
        size_t dim,
        size_t k,
        elips::Metric metric) {
        (void)query; (void)candidate_vectors; (void)list_offsets;
        (void)indices_out; (void)values_out; (void)nlists; (void)ncand;
        (void)dim; (void)k; (void)metric;
        return std::unexpected(GpuError::BackendUnavailable);
    }

    virtual void synchronize() = 0;
    [[nodiscard]] virtual bool is_idle() const noexcept = 0;
};

using GpuPort [[deprecated("renamed to GpuBackend")]] = GpuBackend;

} // namespace elips::gpu

#endif
