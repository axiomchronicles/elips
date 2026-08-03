#include "elips/gpu_engine/GpuDeviceManager.hpp"

#include <algorithm>
#include <string_view>

#include "elips/gpu_engine/GpuSelector.hpp"

#if defined(__APPLE__) && defined(ELIPS_METAL_ENABLED)
#include "MetalBackend.hpp"
#endif

#if defined(ELIPS_CUDA_ENABLED)
#include "cuda/CudaBackend.hpp"
#endif

#if defined(ELIPS_HIP_ENABLED)
#include "hip/HipBackend.hpp"
#endif

#if defined(ELIPS_SYCL_ENABLED)
#include "sycl/SyclBackend.hpp"
#endif

#if defined(ELIPS_VULKAN_ENABLED)
#include "vulkan/VulkanBackend.hpp"
#endif

namespace elips::gpu {
namespace {

GpuDeviceInfo make_cpu_fallback_info() {
    GpuDeviceInfo info;
    info.name = "CPU (SIMD)";
    info.vendor = "CPU";
    info.backend = "cpu";
    info.device_index = 0;
    info.total_device_memory_bytes = 0;
    info.has_unified_memory = true;
    info.supports_cagra = false;
    info.supports_ivf_pq = false;
    return info;
}

} // namespace

GpuDeviceInfo GpuDeviceManager::cpu_fallback_info() const noexcept {
    return make_cpu_fallback_info();
}

GpuDeviceInfo GpuDeviceManager::runtime_device_info() const {
    const auto devices = probe_all_devices();
    if (devices.empty()) {
        return cpu_fallback_info();
    }

    const GpuSelector selector;
    const GpuConfig config;
    auto selected = selector.select(config, devices);
    if (!selected.has_value() || *selected == nullptr) {
        return cpu_fallback_info();
    }

    const GpuDeviceInfo info = (*selected)->device_info();
    (*selected)->shutdown();
    return info;
}

std::vector<GpuDeviceInfo> GpuDeviceManager::probe_all_devices() const {
    std::vector<GpuDeviceInfo> devices;

#if defined(__APPLE__) && defined(ELIPS_METAL_ENABLED)
    try {
        metal::MetalBackend probe;
        auto result = probe.initialize(GpuConfig{});
        if (result.has_value()) {
            devices.push_back(probe.device_info());
            probe.shutdown();
        }
    } catch (...) {}
#endif

#if defined(ELIPS_CUDA_ENABLED)
    try {
        cuda::CudaBackend probe;
        auto result = probe.initialize(GpuConfig{});
        if (result.has_value()) {
            devices.push_back(probe.device_info());
            probe.shutdown();
        }
    } catch (...) {}
#endif

#if defined(ELIPS_HIP_ENABLED)
    try {
        hip::HipBackend probe;
        auto result = probe.initialize(GpuConfig{});
        if (result.has_value()) {
            devices.push_back(probe.device_info());
            probe.shutdown();
        }
    } catch (...) {}
#endif

#if defined(ELIPS_SYCL_ENABLED)
    try {
        sycl::SyclBackend probe;
        auto result = probe.initialize(GpuConfig{});
        if (result.has_value()) {
            devices.push_back(probe.device_info());
            probe.shutdown();
        }
    } catch (...) {}
#endif

#if defined(ELIPS_VULKAN_ENABLED)
    try {
        vulkan::VulkanBackend probe;
        auto result = probe.initialize(GpuConfig{});
        if (result.has_value()) {
            devices.push_back(probe.device_info());
            probe.shutdown();
        }
    } catch (...) {}
#endif

    std::sort(devices.begin(), devices.end(),
              [](const GpuDeviceInfo& a, const GpuDeviceInfo& b) {
                  if (a.supports_cagra != b.supports_cagra) return a.supports_cagra;
                  return a.total_device_memory_bytes > b.total_device_memory_bytes;
              });
    return devices;
}

std::optional<std::unique_ptr<GpuPort>>
GpuDeviceManager::select(const GpuConfig& config,
                         const std::vector<GpuDeviceInfo>& devices) const {
    if (config.policy == GpuPolicy::CpuOnly) {
        return std::nullopt;
    }

    if (devices.empty()) {
        if (config.policy == GpuPolicy::RequireGpu) {
            return std::nullopt;
        }
        return std::nullopt;
    }

    const GpuDeviceInfo* chosen = nullptr;
    for (const auto& candidate : devices) {
        if (!config.preferred_backend.empty() &&
            candidate.backend != config.preferred_backend) {
            continue;
        }
        if (config.device_index >= 0 &&
            candidate.device_index !=
                static_cast<uint32_t>(config.device_index)) {
            continue;
        }
        chosen = &candidate;
        break;
    }

    if (chosen == nullptr) {
        if (config.policy == GpuPolicy::RequireGpu ||
            config.policy == GpuPolicy::Specific) {
            return std::nullopt;
        }
        chosen = &devices.front();
    }

#if defined(ELIPS_CUDA_ENABLED)
    if (chosen->backend == "cuda") {
        auto backend = std::make_unique<cuda::CudaBackend>();
        auto result = backend->initialize(config);
        if (result.has_value()) {
            return backend;
        }
    }
#endif

#if defined(ELIPS_HIP_ENABLED)
    if (chosen->backend == "hip") {
        auto backend = std::make_unique<hip::HipBackend>();
        auto result = backend->initialize(config);
        if (result.has_value()) {
            return backend;
        }
    }
#endif

#if defined(__APPLE__) && defined(ELIPS_METAL_ENABLED)
    if (chosen->backend == "metal") {
        auto backend = std::make_unique<metal::MetalBackend>();
        auto result = backend->initialize(config);
        if (result.has_value()) {
            return backend;
        }
    }
#endif

#if defined(ELIPS_SYCL_ENABLED)
    if (chosen->backend == "sycl") {
        auto backend = std::make_unique<sycl::SyclBackend>();
        auto result = backend->initialize(config);
        if (result.has_value()) {
            return backend;
        }
    }
#endif

#if defined(ELIPS_VULKAN_ENABLED)
    if (chosen->backend == "vulkan") {
        auto backend = std::make_unique<vulkan::VulkanBackend>();
        auto result = backend->initialize(config);
        if (result.has_value()) {
            return backend;
        }
    }
#endif

    if (config.policy == GpuPolicy::RequireGpu) {
        return std::nullopt;
    }
    return std::nullopt;
}

bool GpuDeviceManager::can_fit_index(
    const GpuDeviceInfo& dev, size_t n_vectors, size_t dim,
    const GpuConfig& config) const {
    size_t required = n_vectors * dim * sizeof(float);
    if (config.algorithm == GpuIndexAlgorithm::CagraGraph) {
        required += n_vectors * config.graph_params.graph_degree * sizeof(uint32_t);
        required += n_vectors * sizeof(int);
    }
    return dev.free_device_memory_bytes > required;
}

} // namespace elips::gpu
