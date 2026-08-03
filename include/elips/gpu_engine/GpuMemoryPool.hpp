#ifndef ELIPS_GPU_ENGINE_GPU_MEMORY_POOL_HPP
#define ELIPS_GPU_ENGINE_GPU_MEMORY_POOL_HPP

#include <cstddef>
#include <mutex>
#include <vector>

#include "elips/domain/Expected.hpp"
#include "elips/gpu_engine/GpuBuffer.hpp"
#include "elips/gpu_engine/GpuPort.hpp"

namespace elips::gpu {

class GpuMemoryPool {
public:
    GpuMemoryPool(GpuPort& backend, size_t total_bytes);
    ~GpuMemoryPool();

    GpuMemoryPool(const GpuMemoryPool&) = delete;
    GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;
    GpuMemoryPool(GpuMemoryPool&&) = delete;
    GpuMemoryPool& operator=(GpuMemoryPool&&) = delete;

    [[nodiscard]] std::expected<GpuBuffer, GpuError> acquire(size_t bytes, size_t alignment = 256);
    void release(GpuBuffer&& buf) noexcept;

    [[nodiscard]] size_t available() const noexcept;
    [[nodiscard]] size_t used() const noexcept;

private:
    struct Block {
        void* ptr;
        size_t bytes;
        bool free;
    };

    GpuPort& backend_;
    GpuBuffer pool_buffer_;
    size_t total_bytes_;
    std::vector<Block> blocks_;
    mutable std::mutex mutex_;
};

} // namespace elips::gpu

#endif