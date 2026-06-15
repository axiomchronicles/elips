#ifndef ELIPS_GPU_ENGINE_GPU_DEVICE_INFO_HPP
#define ELIPS_GPU_ENGINE_GPU_DEVICE_INFO_HPP

#include <cstdint>
#include <string>

namespace elips::gpu {

struct GpuDeviceInfo {
    std::string name;
    std::string vendor;
    std::string backend;
    uint32_t device_index{0};

    size_t total_device_memory_bytes{0};
    size_t free_device_memory_bytes{0};
    bool has_unified_memory{false};
    bool supports_fp16{false};
    bool supports_bf16{false};
    bool supports_int8{false};

    uint32_t compute_capability_major{0};
    uint32_t compute_capability_minor{0};
    uint32_t max_threads_per_block{0};
    uint32_t multiprocessor_count{0};
    size_t shared_memory_per_block_bytes{0};
    size_t l2_cache_bytes{0};
    float peak_tflops_fp32{0.0f};
    float peak_tflops_fp16{0.0f};

    float host_to_device_bandwidth_gb_s{0.0f};
    float device_to_host_bandwidth_gb_s{0.0f};

    bool supports_cagra{false};
    bool supports_ivf_pq{false};
    bool supports_dynamic_batching{false};
    bool supports_half_precision_search{false};
};

} // namespace elips::gpu

#endif