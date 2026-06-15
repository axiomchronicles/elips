#ifndef ELIPS_GPU_ENGINE_GPU_BUFFER_HPP
#define ELIPS_GPU_ENGINE_GPU_BUFFER_HPP

#include <cstddef>
#include <utility>

namespace elips::gpu {

class GpuBuffer {
public:
    GpuBuffer() = default;

    GpuBuffer(void* device_ptr, size_t bytes, void* backend_handle)
        : device_ptr_(device_ptr), bytes_(bytes), backend_handle_(backend_handle) {}

    ~GpuBuffer() = default;

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    GpuBuffer(GpuBuffer&& other) noexcept
        : device_ptr_(other.device_ptr_)
        , bytes_(other.bytes_)
        , backend_handle_(other.backend_handle_) {
        other.device_ptr_ = nullptr;
        other.bytes_ = 0;
        other.backend_handle_ = nullptr;
    }

    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            device_ptr_ = other.device_ptr_;
            bytes_ = other.bytes_;
            backend_handle_ = other.backend_handle_;
            other.device_ptr_ = nullptr;
            other.bytes_ = 0;
            other.backend_handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] void* device_ptr() const noexcept { return device_ptr_; }
    [[nodiscard]] size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] void* backend_handle() const noexcept { return backend_handle_; }
    [[nodiscard]] explicit operator bool() const noexcept { return device_ptr_ != nullptr; }

private:
    void* device_ptr_{nullptr};
    size_t bytes_{0};
    void* backend_handle_{nullptr};
};

} // namespace elips::gpu

#endif