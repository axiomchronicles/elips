#ifndef ELIPS_GPU_ENGINE_GPU_PROFILER_HPP
#define ELIPS_GPU_ENGINE_GPU_PROFILER_HPP

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

namespace elips::gpu {

struct KernelTiming {
    std::string kernel_name;
    std::chrono::microseconds duration;
    size_t work_items;
};

class GpuProfiler {
public:
    void record(const std::string& kernel, std::chrono::microseconds duration, size_t items = 0);
    [[nodiscard]] std::vector<KernelTiming> recent_timings(size_t max_count = 100) const;
    [[nodiscard]] size_t total_launches() const noexcept;
    void clear();

private:
    std::vector<KernelTiming> timings_;
    std::atomic<size_t> total_launches_{0};
};

} // namespace elips::gpu

#endif