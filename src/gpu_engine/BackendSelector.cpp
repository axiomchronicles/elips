#include "elips/gpu_engine/BackendSelector.hpp"
#include "elips/gpu_engine/GpuDeviceManager.hpp"

#include <algorithm>

namespace elips::gpu {

int BackendSelector::rank_backend(std::string_view backend) const noexcept {
    if (backend == "cuda")   return 100;
    if (backend == "hip")    return 90;
    if (backend == "metal")  return 80;
    if (backend == "cpu")    return 0;
    return 0;
}

std::optional<std::unique_ptr<GpuBackend>>
BackendSelector::select(const GpuConfig& config,
                    const std::vector<GpuDeviceInfo>& devices) const {
    if (config.policy == GpuPolicy::CpuOnly) return std::nullopt;
    if (devices.empty()) {
        if (config.policy == GpuPolicy::RequireGpu) return std::nullopt;
        return std::nullopt;
    }

    std::vector<GpuDeviceInfo> ordered = devices;
    if (config.policy == GpuPolicy::Specific) {
        ordered.erase(
            std::remove_if(ordered.begin(), ordered.end(),
                           [&](const GpuDeviceInfo& dev) {
                               if (!config.preferred_backend.empty() &&
                                   dev.backend != config.preferred_backend) {
                                   return true;
                               }
                               if (config.device_index >= 0 &&
                                   dev.device_index !=
                                       static_cast<uint32_t>(config.device_index)) {
                                   return true;
                               }
                               return false;
                           }),
            ordered.end());
        if (ordered.empty()) {
            return std::nullopt;
        }
    }

    std::stable_sort(
        ordered.begin(), ordered.end(),
        [&](const GpuDeviceInfo& a, const GpuDeviceInfo& b) {
            const int rank_a = rank_backend(a.backend);
            const int rank_b = rank_backend(b.backend);
            if (rank_a != rank_b) {
                return rank_a > rank_b;
            }
            if (a.supports_cagra != b.supports_cagra) {
                return a.supports_cagra > b.supports_cagra;
            }
            return a.total_device_memory_bytes > b.total_device_memory_bytes;
        });

    GpuDeviceManager manager;
    return manager.select(config, ordered);
}

} // namespace elips::gpu
