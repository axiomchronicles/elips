#ifndef ELIPS_GPU_ENGINE_UNIFIED_BUFFER_HPP
#define ELIPS_GPU_ENGINE_UNIFIED_BUFFER_HPP

#include <cstddef>

#include "elips/gpu_engine/GpuBuffer.hpp"

namespace elips::gpu {

class UnifiedBuffer {
public:
    UnifiedBuffer() = default;

    UnifiedBuffer(void* ptr, size_t bytes, GpuBuffer gpu_view)
        : ptr_(ptr), bytes_(bytes), gpu_view_(std::move(gpu_view)) {}

    [[nodiscard]] void* host_ptr() noexcept { return ptr_; }
    [[nodiscard]] const void* host_ptr() const noexcept { return ptr_; }
    [[nodiscard]] size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] const GpuBuffer& gpu_view() const noexcept { return gpu_view_; }
    [[nodiscard]] GpuBuffer& gpu_view() noexcept { return gpu_view_; }

private:
    void* ptr_{nullptr};
    size_t bytes_{0};
    GpuBuffer gpu_view_;
};

} // namespace elips::gpu

#endif