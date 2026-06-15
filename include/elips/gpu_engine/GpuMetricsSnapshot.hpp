#ifndef ELIPS_GPU_ENGINE_GPU_METRICS_SNAPSHOT_HPP
#define ELIPS_GPU_ENGINE_GPU_METRICS_SNAPSHOT_HPP

#include <cstddef>
#include <string>

namespace elips::gpu {

struct GpuMetricsSnapshot {
    std::string backend;
    std::string device_name;
    size_t device_memory_used_bytes{0};
    size_t device_memory_total_bytes{0};
    size_t index_build_count{0};
    size_t index_build_time_total_ms{0};
    float index_build_speedup_vs_cpu_avg{0.0f};
    size_t search_kernel_launches_total{0};
    size_t search_p50_latency_us{0};
    size_t search_p99_latency_us{0};
    float batch_avg_size{0.0f};
    float batch_coalescing_ratio{0.0f};
    bool fp16_search_enabled{false};
    size_t fallback_events_total{0};
    size_t kernel_errors_total{0};
    size_t pinned_memory_pool_used_bytes{0};
};

} // namespace elips::gpu

#endif