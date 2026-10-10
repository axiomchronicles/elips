#include "elips/gpu_engine/GpuMemoryPool.hpp"
#include <algorithm>

namespace elips::gpu {

GpuMemoryPool::GpuMemoryPool(GpuBackend& backend, size_t total_bytes)
    : backend_(backend), total_bytes_(total_bytes) {
    auto alloc = backend_.allocate_device(total_bytes);
    if (alloc.has_value()) {
        pool_buffer_ = std::move(*alloc);
        blocks_.push_back({pool_buffer_.device_ptr(), total_bytes, true});
    }
}

GpuMemoryPool::~GpuMemoryPool() {
    if (pool_buffer_) {
        backend_.free_device(std::move(pool_buffer_));
    }
}

std::expected<GpuBuffer, GpuError> GpuMemoryPool::acquire(size_t bytes, size_t alignment) {
    std::lock_guard lock(mutex_);
    if (!pool_buffer_) {
        return backend_.allocate_device(bytes);
    }

    for (auto& block : blocks_) {
        if (block.free && block.bytes >= bytes) {
            block.free = false;
            if (block.bytes > bytes + alignment) {
                Block remaining;
                remaining.ptr = static_cast<char*>(block.ptr) + bytes;
                remaining.bytes = block.bytes - bytes;
                remaining.free = true;
                blocks_.push_back(remaining);
                block.bytes = bytes;
            }
            return GpuBuffer{block.ptr, bytes, nullptr};
        }
    }
    return std::unexpected(GpuError::InsufficientMemory);
}

void GpuMemoryPool::release(GpuBuffer&& buf) noexcept {
    if (!buf) return;
    std::lock_guard lock(mutex_);
    for (auto& block : blocks_) {
        if (block.ptr == buf.device_ptr() && !block.free) {
            block.free = true;

            auto next = std::find_if(blocks_.begin(), blocks_.end(),
                                     [&](const Block& b) {
                                         return b.ptr == static_cast<char*>(block.ptr) + block.bytes;
                                     });
            if (next != blocks_.end() && next->free) {
                block.bytes += next->bytes;
                blocks_.erase(next);
            }
            return;
        }
    }
    if (buf.backend_handle()) {
        backend_.free_device(std::move(buf));
    }
}

size_t GpuMemoryPool::available() const noexcept {
    std::lock_guard lock(mutex_);
    size_t avail = 0;
    for (const auto& block : blocks_) {
        if (block.free) avail += block.bytes;
    }
    return avail;
}

size_t GpuMemoryPool::used() const noexcept {
    std::lock_guard lock(mutex_);
    return total_bytes_ - available();
}

} // namespace elips::gpu
