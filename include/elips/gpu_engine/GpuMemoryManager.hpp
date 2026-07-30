#ifndef ELIPS_GPU_ENGINE_GPU_MEMORY_MANAGER_HPP
#define ELIPS_GPU_ENGINE_GPU_MEMORY_MANAGER_HPP

#include <cstddef>
#include <expected>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "elips/gpu_engine/GpuBuffer.hpp"
#include "elips/gpu_engine/GpuMemoryPort.hpp"
#include "elips/gpu_engine/GpuPort.hpp"
#include "elips/gpu_engine/PinnedBuffer.hpp"

namespace elips::gpu {

// A span of device memory available for suballocation. `root` identifies which
// backend allocation it came from; blocks from different roots are not
// contiguous and must never be merged.
struct FreeBlock {
    void* ptr;
    size_t bytes;
    size_t root;
};

class GpuMemoryManager : public GpuMemoryPort {
public:
    explicit GpuMemoryManager(GpuPort& backend);
    ~GpuMemoryManager() override;

    GpuMemoryManager(const GpuMemoryManager&) = delete;
    GpuMemoryManager& operator=(const GpuMemoryManager&) = delete;
    GpuMemoryManager(GpuMemoryManager&&) = delete;
    GpuMemoryManager& operator=(GpuMemoryManager&&) = delete;

    [[nodiscard]] std::expected<void, GpuError>
    initialize(size_t pool_bytes);

    [[nodiscard]] std::expected<GpuBuffer, GpuError>
    allocate(size_t bytes, size_t alignment = 256) override;

    void deallocate(GpuBuffer&& buf) noexcept override;

    [[nodiscard]] std::expected<void*, GpuError>
    allocate_pinned(size_t bytes) override;

    void deallocate_pinned(void* ptr) noexcept override;

    [[nodiscard]] size_t bytes_used() const noexcept override;
    [[nodiscard]] size_t bytes_available() const noexcept override;
    [[nodiscard]] size_t peak_bytes_used() const noexcept override;

    void shutdown() noexcept;

private:
    struct LiveBlock {
        size_t bytes;  // rounded-up size actually reserved
        size_t root;
    };

    // Return a span to the free list, merging it with any physically adjacent
    // free block from the same root allocation.
    void release_locked(void* ptr, size_t bytes, size_t root) noexcept;

    GpuPort& backend_;
    size_t pool_bytes_{0};      // ceiling we are allowed to request from backend
    size_t pool_committed_{0};  // bytes actually obtained from backend so far
    size_t allocated_{0};       // bytes handed out to callers (rounded)
    size_t peak_allocated_{0};
    std::vector<FreeBlock> free_blocks_;
    std::vector<GpuBuffer> root_allocations_;
    std::unordered_map<void*, LiveBlock> live_blocks_;
    std::vector<void*> pinned_blocks_;
    mutable std::mutex mutex_;
};

} // namespace elips::gpu

#endif
