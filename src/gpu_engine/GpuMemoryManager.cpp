#include "elips/gpu_engine/GpuMemoryManager.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace elips::gpu {
namespace {

[[nodiscard]] size_t round_up(size_t value, size_t alignment) noexcept {
    if (alignment == 0) {
        return value;
    }
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

} // namespace

GpuMemoryManager::GpuMemoryManager(GpuPort& backend)
    : backend_(backend) {}

GpuMemoryManager::~GpuMemoryManager() {
    shutdown();
}

std::expected<void, GpuError> GpuMemoryManager::initialize(size_t pool_bytes) {
    if (pool_bytes == 0) {
        auto info = backend_.device_info();
        pool_bytes = static_cast<size_t>(static_cast<double>(info.total_device_memory_bytes) * 0.8);
    }
    pool_bytes_ = pool_bytes;
    return {};
}

std::expected<GpuBuffer, GpuError>
GpuMemoryManager::allocate(size_t bytes, size_t alignment) {
    std::lock_guard lock(mutex_);
    // Reserve in aligned units so every split leaves an aligned remainder. A
    // zero-byte request still reserves one unit, so the returned pointer is
    // unique and round-trips through deallocate().
    const size_t want = round_up(std::max<size_t>(bytes, 1), alignment == 0 ? 1 : alignment);

    size_t best_idx = SIZE_MAX;
    size_t best_waste = SIZE_MAX;
    for (size_t i = 0; i < free_blocks_.size(); ++i) {
        if (free_blocks_[i].bytes >= want) {
            const size_t waste = free_blocks_[i].bytes - want;
            if (waste < best_waste) {
                best_waste = waste;
                best_idx = i;
            }
        }
    }

    if (best_idx != SIZE_MAX) {
        const FreeBlock block = free_blocks_[best_idx];
        free_blocks_.erase(free_blocks_.begin() + static_cast<ptrdiff_t>(best_idx));
        // Split: the tail of an oversized block goes back on the free list.
        // Dropping it here would leak those bytes for the pool's lifetime.
        if (block.bytes > want) {
            free_blocks_.push_back({static_cast<char*>(block.ptr) + want,
                                    block.bytes - want, block.root});
        }
        allocated_ += want;
        peak_allocated_ = std::max(peak_allocated_, allocated_);
        live_blocks_[block.ptr] = LiveBlock{want, block.root};
        return GpuBuffer{block.ptr, bytes, nullptr};
    }

    const size_t alloc_size = std::max(want, pool_bytes_ / 16);
    if (pool_committed_ + alloc_size > pool_bytes_) {
        return std::unexpected(GpuError::InsufficientMemory);
    }

    auto result = backend_.allocate_device(alloc_size);
    if (!result.has_value()) return std::unexpected(result.error());

    void* base_ptr = result->device_ptr();
    const size_t base_bytes = result->bytes();
    if (base_bytes < want) {
        backend_.free_device(std::move(*result));
        return std::unexpected(GpuError::InsufficientMemory);
    }
    const size_t root = root_allocations_.size();
    root_allocations_.push_back(std::move(*result));
    pool_committed_ += base_bytes;

    allocated_ += want;
    peak_allocated_ = std::max(peak_allocated_, allocated_);
    live_blocks_[base_ptr] = LiveBlock{want, root};

    if (base_bytes > want) {
        free_blocks_.push_back({static_cast<char*>(base_ptr) + want,
                                base_bytes - want, root});
    }

    return GpuBuffer{base_ptr, bytes, nullptr};
}

void GpuMemoryManager::release_locked(void* ptr, size_t bytes,
                                      size_t root) noexcept {
    // Coalesce with physically adjacent free blocks from the same root
    // allocation. Without this the free list fragments into pieces that never
    // re-merge, so a workload with enough total free bytes still fails.
    char* begin = static_cast<char*>(ptr);
    char* end = begin + bytes;
    for (size_t i = free_blocks_.size(); i-- > 0;) {
        const FreeBlock& block = free_blocks_[i];
        if (block.root != root) {
            continue;
        }
        char* block_begin = static_cast<char*>(block.ptr);
        char* block_end = block_begin + block.bytes;
        if (block_end == begin) {
            begin = block_begin;
        } else if (block_begin == end) {
            end = block_end;
        } else {
            continue;
        }
        free_blocks_.erase(free_blocks_.begin() + static_cast<ptrdiff_t>(i));
    }
    free_blocks_.push_back({begin, static_cast<size_t>(end - begin), root});
}

void GpuMemoryManager::deallocate(GpuBuffer&& buf) noexcept {
    if (!buf) return;
    std::lock_guard lock(mutex_);
    const auto it = live_blocks_.find(buf.device_ptr());
    if (it == live_blocks_.end()) {
        return;  // not ours, or already freed: dropping is safer than corrupting
    }
    const LiveBlock block = it->second;
    live_blocks_.erase(it);
    allocated_ -= block.bytes;
    release_locked(buf.device_ptr(), block.bytes, block.root);
}

std::expected<void*, GpuError> GpuMemoryManager::allocate_pinned(size_t bytes) {
    constexpr size_t alignment = 4096;
    const size_t rounded = round_up(bytes, alignment);
    void* ptr = std::aligned_alloc(alignment, rounded);
    if (!ptr) return std::unexpected(GpuError::InsufficientMemory);
    std::lock_guard lock(mutex_);
    pinned_blocks_.push_back(ptr);
    return ptr;
}

void GpuMemoryManager::deallocate_pinned(void* ptr) noexcept {
    if (!ptr) return;
    std::lock_guard lock(mutex_);
    auto it = std::find(pinned_blocks_.begin(), pinned_blocks_.end(), ptr);
    if (it != pinned_blocks_.end()) {
        pinned_blocks_.erase(it);
    }
    std::free(ptr);
}

size_t GpuMemoryManager::bytes_used() const noexcept {
    std::lock_guard lock(mutex_);
    return allocated_;
}

size_t GpuMemoryManager::bytes_available() const noexcept {
    std::lock_guard lock(mutex_);
    // Bytes a caller could still obtain: what is on the free list plus what the
    // pool ceiling still allows us to request from the backend. Reporting
    // pool_bytes_ - allocated_ would over-report, since committed-but-fragmented
    // bytes are not all reachable.
    size_t free_bytes = 0;
    for (const auto& block : free_blocks_) {
        free_bytes += block.bytes;
    }
    return free_bytes + (pool_bytes_ - pool_committed_);
}

size_t GpuMemoryManager::peak_bytes_used() const noexcept {
    std::lock_guard lock(mutex_);
    return peak_allocated_;
}

void GpuMemoryManager::shutdown() noexcept {
    std::lock_guard lock(mutex_);
    for (auto& allocation : root_allocations_) {
        backend_.free_device(std::move(allocation));
    }
    free_blocks_.clear();
    root_allocations_.clear();
    live_blocks_.clear();
    for (auto* ptr : pinned_blocks_) {
        std::free(ptr);
    }
    pinned_blocks_.clear();
    pool_bytes_ = 0;
    pool_committed_ = 0;
    allocated_ = 0;
}

} // namespace elips::gpu
