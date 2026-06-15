#ifndef ELIPS_GPU_ENGINE_PINNED_BUFFER_HPP
#define ELIPS_GPU_ENGINE_PINNED_BUFFER_HPP

#include <cstddef>
#include <utility>

namespace elips::gpu {

class PinnedBuffer {
public:
    PinnedBuffer() = default;

    PinnedBuffer(void* data, size_t bytes)
        : data_(data), bytes_(bytes) {}

    ~PinnedBuffer() = default;

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept
        : data_(other.data_), bytes_(other.bytes_) {
        other.data_ = nullptr;
        other.bytes_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            data_ = other.data_;
            bytes_ = other.bytes_;
            other.data_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] explicit operator bool() const noexcept { return data_ != nullptr; }

private:
    void* data_{nullptr};
    size_t bytes_{0};
};

} // namespace elips::gpu

#endif