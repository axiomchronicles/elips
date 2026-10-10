#ifndef ELIPS_GPU_ENGINE_BACKEND_SELECTOR_HPP
#define ELIPS_GPU_ENGINE_BACKEND_SELECTOR_HPP

#include <memory>
#include <optional>
#include <vector>

#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuDeviceInfo.hpp"
#include "elips/gpu_engine/GpuBackend.hpp"

namespace elips::gpu {

class BackendSelector {
public:
    [[nodiscard]] std::optional<std::unique_ptr<GpuBackend>>
    select(const GpuConfig& config, const std::vector<GpuDeviceInfo>& devices) const;

private:
    [[nodiscard]] int rank_backend(std::string_view backend) const noexcept;
};

using GpuSelector [[deprecated("renamed to BackendSelector")]] = BackendSelector;

} // namespace elips::gpu

#endif