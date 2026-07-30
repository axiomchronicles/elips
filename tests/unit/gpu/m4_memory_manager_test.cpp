// M4 regression tests: GPU suballocator accounting, leak-freedom, and free-list
// coalescing (F9). Uses a host-memory fake backend so these run on any machine,
// with or without a real GPU.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

#include "elips/gpu_engine/GpuMemoryManager.hpp"
#include "elips/gpu_engine/GpuPort.hpp"

namespace elips::gpu {
namespace {

// Minimal GpuPort that hands out real host memory, so pointer arithmetic in the
// suballocator (splitting, adjacency, coalescing) is exercised for real.
class FakeBackend final : public GpuPort {
public:
    ~FakeBackend() override {
        for (auto& [ptr, bytes] : owned_) {
            (void)bytes;
            std::free(ptr);
        }
    }

    std::expected<void, GpuError> initialize(const GpuConfig&) override {
        return {};
    }
    void shutdown() noexcept override {}

    GpuDeviceInfo device_info() const noexcept override {
        GpuDeviceInfo info;
        info.total_device_memory_bytes = 1ULL << 30;
        return info;
    }
    bool is_available() const noexcept override { return true; }

    std::expected<GpuBuffer, GpuError> allocate_device(size_t bytes) override {
        void* ptr = std::aligned_alloc(4096, ((bytes + 4095) / 4096) * 4096);
        if (ptr == nullptr) {
            return std::unexpected(GpuError::InsufficientMemory);
        }
        owned_[ptr] = bytes;
        ++backend_allocations;
        return GpuBuffer{ptr, bytes, nullptr};
    }

    void free_device(GpuBuffer buf) noexcept override {
        const auto it = owned_.find(buf.device_ptr());
        if (it == owned_.end()) {
            return;
        }
        std::free(it->first);
        owned_.erase(it);
    }

    std::expected<void, GpuError> upload(const void*, GpuBuffer&,
                                        size_t) override {
        return {};
    }
    std::expected<void, GpuError> download(const GpuBuffer&, void*,
                                           size_t) override {
        return {};
    }
    std::expected<void, GpuError> compute_distances_batch(
        std::span<const float>, std::span<const float>, std::span<float>, size_t,
        size_t, size_t, elips::Metric) override {
        return {};
    }
    std::expected<void, GpuError> top_k(std::span<const float>,
                                        std::span<uint32_t>, std::span<float>,
                                        size_t, size_t, size_t) override {
        return {};
    }
    void synchronize() override {}
    bool is_idle() const noexcept override { return true; }

    [[nodiscard]] size_t live_backend_allocations() const { return owned_.size(); }

    size_t backend_allocations{0};

private:
    std::map<void*, size_t> owned_;
};

constexpr size_t kPool = 16UL * 1024 * 1024;

TEST(M4GpuMemory, ReusedBlockSplitDoesNotLeakRemainder) {
    FakeBackend backend;
    GpuMemoryManager pool(backend);
    ASSERT_TRUE(pool.initialize(kPool).has_value());

    // Carve out a large block, free it, then take a small slice of it. The
    // remainder of the reused block must return to the free list, not vanish.
    auto big = pool.allocate(1024 * 1024);
    ASSERT_TRUE(big.has_value());
    pool.deallocate(std::move(*big));

    const size_t available_before = pool.bytes_available();
    auto small = pool.allocate(4096);
    ASSERT_TRUE(small.has_value());
    EXPECT_EQ(pool.bytes_used(), 4096U);
    // Only the 4 KiB handed out should have left the available pool.
    EXPECT_EQ(pool.bytes_available(), available_before - 4096U);

    pool.deallocate(std::move(*small));
    EXPECT_EQ(pool.bytes_used(), 0U);
    EXPECT_EQ(pool.bytes_available(), available_before);
}

TEST(M4GpuMemory, AccountingReturnsToZeroAfterChurn) {
    FakeBackend backend;
    GpuMemoryManager pool(backend);
    ASSERT_TRUE(pool.initialize(kPool).has_value());
    const size_t initial_available = pool.bytes_available();

    std::mt19937 rng(2024);
    std::uniform_int_distribution<size_t> size_dist(256, 256 * 1024);
    std::vector<GpuBuffer> held;

    for (int round = 0; round < 500; ++round) {
        if (held.size() < 8 || (rng() % 2) == 0) {
            auto buf = pool.allocate(size_dist(rng));
            if (buf.has_value()) {
                held.push_back(std::move(*buf));
            }
        } else {
            const size_t idx = rng() % held.size();
            pool.deallocate(std::move(held[idx]));
            held.erase(held.begin() + static_cast<std::ptrdiff_t>(idx));
        }
        // bytes_used never exceeds what the pool could possibly hold.
        EXPECT_LE(pool.bytes_used(), kPool);
    }

    for (auto& buf : held) {
        pool.deallocate(std::move(buf));
    }
    held.clear();

    // No drift: every byte handed out came back.
    EXPECT_EQ(pool.bytes_used(), 0U);
    EXPECT_EQ(pool.bytes_available(), initial_available);
}

TEST(M4GpuMemory, FragmentationDoesNotCauseSpuriousFailure) {
    FakeBackend backend;
    GpuMemoryManager pool(backend);
    ASSERT_TRUE(pool.initialize(kPool).has_value());

    // Fill with many small blocks, free them all, then ask for a block that
    // only fits if the free list coalesced back into a contiguous span. Blocks
    // from different backend allocations are genuinely not adjacent, so the
    // request is sized to one root allocation (pool/16).
    constexpr size_t kChunk = 64 * 1024;
    constexpr size_t kRootBytes = kPool / 16;
    constexpr int kChunks = 32;
    std::vector<GpuBuffer> chunks;
    for (int i = 0; i < kChunks; ++i) {
        auto buf = pool.allocate(kChunk);
        ASSERT_TRUE(buf.has_value()) << "failed at chunk " << i;
        chunks.push_back(std::move(*buf));
    }
    // Free in an interleaved order so merging has to happen from both sides.
    for (std::size_t i = 0; i < chunks.size(); i += 2) {
        pool.deallocate(std::move(chunks[i]));
    }
    for (std::size_t i = 1; i < chunks.size(); i += 2) {
        pool.deallocate(std::move(chunks[i]));
    }
    chunks.clear();
    EXPECT_EQ(pool.bytes_used(), 0U);

    // Without coalescing the free list would hold 32 separate 64 KiB pieces and
    // this request would spuriously fail despite ample free bytes.
    const size_t backend_allocations_before = backend.backend_allocations;
    auto big = pool.allocate(kRootBytes);
    ASSERT_TRUE(big.has_value())
        << "coalescing failed: " << pool.bytes_available()
        << " bytes free but a " << kRootBytes << " byte request was refused";
    // Served from the merged free list, not by growing the pool again.
    EXPECT_EQ(backend.backend_allocations, backend_allocations_before);
    pool.deallocate(std::move(*big));
}

TEST(M4GpuMemory, AvailableBytesNeverOverReport) {
    FakeBackend backend;
    GpuMemoryManager pool(backend);
    ASSERT_TRUE(pool.initialize(kPool).has_value());

    std::vector<GpuBuffer> held;
    // Drain the pool. Every successful allocation must have been promised by
    // bytes_available() beforehand.
    while (true) {
        const size_t available = pool.bytes_available();
        auto buf = pool.allocate(512 * 1024);
        if (!buf.has_value()) {
            break;
        }
        EXPECT_GE(available, 512U * 1024U)
            << "bytes_available() under-reported before a successful allocation";
        held.push_back(std::move(*buf));
    }
    EXPECT_LT(pool.bytes_available(), 512U * 1024U)
        << "pool refused an allocation while claiming enough bytes were free";

    for (auto& buf : held) {
        pool.deallocate(std::move(buf));
    }
}

TEST(M4GpuMemory, DoubleFreeDoesNotCorruptAccounting) {
    FakeBackend backend;
    GpuMemoryManager pool(backend);
    ASSERT_TRUE(pool.initialize(kPool).has_value());

    auto buf = pool.allocate(8192);
    ASSERT_TRUE(buf.has_value());
    void* ptr = buf->device_ptr();
    pool.deallocate(std::move(*buf));
    EXPECT_EQ(pool.bytes_used(), 0U);

    // A stale handle naming the same pointer must not double-decrement.
    GpuBuffer stale{ptr, 8192, nullptr};
    pool.deallocate(std::move(stale));
    EXPECT_EQ(pool.bytes_used(), 0U);
}

TEST(M4GpuMemory, ShutdownReleasesEveryBackendAllocation) {
    FakeBackend backend;
    {
        GpuMemoryManager pool(backend);
        ASSERT_TRUE(pool.initialize(kPool).has_value());
        std::vector<GpuBuffer> held;
        for (int i = 0; i < 10; ++i) {
            auto buf = pool.allocate(1024 * 1024);
            if (buf.has_value()) {
                held.push_back(std::move(*buf));
            }
        }
        ASSERT_GT(backend.live_backend_allocations(), 0U);
    }
    EXPECT_EQ(backend.live_backend_allocations(), 0U);
}

}  // namespace
}  // namespace elips::gpu
