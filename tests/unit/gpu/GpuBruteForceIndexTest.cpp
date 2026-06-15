#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/gpu_engine/GpuBruteForceIndex.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu {
namespace {

class RecordingBackend final : public GpuPort {
public:
    [[nodiscard]] std::expected<void, GpuError>
    initialize(const GpuConfig&) override {
        return {};
    }

    void shutdown() noexcept override {}

    [[nodiscard]] GpuDeviceInfo device_info() const noexcept override {
        GpuDeviceInfo info;
        info.name = "recording-backend";
        info.vendor = "test";
        info.backend = "test";
        return info;
    }

    [[nodiscard]] bool is_available() const noexcept override { return true; }

    [[nodiscard]] std::expected<GpuBuffer, GpuError>
    allocate_device(size_t bytes) override {
        return GpuBuffer{nullptr, bytes, this};
    }

    void free_device(GpuBuffer&& buf) noexcept override {
        if (buf.backend_handle() == this) {
            ++free_calls_;
        }
    }

    [[nodiscard]] std::expected<void, GpuError>
    upload(const void* host_src, GpuBuffer&, size_t bytes) override {
        const auto* begin = static_cast<const float*>(host_src);
        uploaded_.assign(begin, begin + bytes / sizeof(float));
        return {};
    }

    [[nodiscard]] std::expected<void, GpuError>
    download(const GpuBuffer&, void* host_dst, size_t bytes) override {
        if (uploaded_.empty()) {
            return {};
        }
        const size_t copy_bytes = std::min(bytes, uploaded_.size() * sizeof(float));
        std::memcpy(host_dst, uploaded_.data(), copy_bytes);
        return {};
    }

    [[nodiscard]] std::expected<void, GpuError>
    compute_distances_batch(std::span<const float> queries,
                            std::span<const float> database,
                            std::span<float> distances_out, size_t nq,
                            size_t nb, size_t dim,
                            elips::Metric metric) override {
        saw_expected_database_ = database.data() != nullptr &&
                                 database.size() == uploaded_.size() &&
                                 std::equal(database.begin(), database.end(),
                                            uploaded_.begin());
        if (!saw_expected_database_) {
            return std::unexpected(GpuError::TransferFailed);
        }

        for (size_t q = 0; q < nq; ++q) {
            const auto query = queries.subspan(q * dim, dim);
            for (size_t i = 0; i < nb; ++i) {
                const auto row = database.subspan(i * dim, dim);
                distances_out[q * nb + i] = elips::distance(metric, query, row);
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GpuError>
    top_k(std::span<const float> distances, std::span<uint32_t> indices_out,
          std::span<float> values_out, size_t nq, size_t nb,
          size_t k) override {
        for (size_t q = 0; q < nq; ++q) {
            std::vector<std::pair<float, uint32_t>> ranked;
            ranked.reserve(nb);
            for (size_t i = 0; i < nb; ++i) {
                ranked.emplace_back(distances[q * nb + i],
                                    static_cast<uint32_t>(i));
            }
            std::partial_sort(ranked.begin(),
                              ranked.begin() + static_cast<ptrdiff_t>(k),
                              ranked.end());
            for (size_t i = 0; i < k; ++i) {
                indices_out[q * k + i] = ranked[i].second;
                values_out[q * k + i] = ranked[i].first;
            }
        }
        return {};
    }

    void synchronize() override {}
    [[nodiscard]] bool is_idle() const noexcept override { return true; }

    [[nodiscard]] bool saw_expected_database() const noexcept {
        return saw_expected_database_;
    }

    [[nodiscard]] size_t free_calls() const noexcept { return free_calls_; }

private:
    size_t free_calls_{0};
    bool saw_expected_database_{false};
    std::vector<float> uploaded_;
};

TEST(GpuBruteForceIndexTest,
     search_uses_host_database_when_device_pointer_is_unreadable) {
    RecordingBackend backend;
    GpuConfig config;

    const RecordID near = RecordID::generate();
    const RecordID far = RecordID::generate();
    const std::vector<float> vectors{0.0F, 0.0F, 10.0F, 10.0F};
    const std::vector<RecordID> ids{near, far};

    {
        GpuBruteForceIndex index(backend, Metric::euclidean, 2, config);
        auto build =
            index.build_from_batch(vectors, ids, GpuIndexBuildParams{});
        ASSERT_TRUE(build.has_value());
        EXPECT_EQ(index.device_bytes_used(), vectors.size() * sizeof(float));

        const auto hits = index.search(std::vector<float>{0.1F, 0.1F}, 2);
        ASSERT_EQ(hits.size(), 2U);
        EXPECT_EQ(hits[0].first, near);
        EXPECT_TRUE(backend.saw_expected_database());
    }

    EXPECT_EQ(backend.free_calls(), 1U);
}

TEST(GpuBruteForceIndexTest, insert_and_remove_keep_search_state_consistent) {
    RecordingBackend backend;
    GpuConfig config;
    GpuBruteForceIndex index(backend, Metric::euclidean, 2, config);

    const RecordID far = RecordID::generate();
    const RecordID near = RecordID::generate();

    index.insert(far, std::vector<float>{10.0F, 10.0F});
    index.insert(near, std::vector<float>{0.0F, 0.0F});

    auto hits = index.search(std::vector<float>{0.1F, 0.1F}, 2);
    ASSERT_EQ(hits.size(), 2U);
    EXPECT_EQ(hits[0].first, near);
    EXPECT_TRUE(backend.saw_expected_database());

    index.remove(near);
    hits = index.search(std::vector<float>{0.1F, 0.1F}, 2);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].first, far);
    EXPECT_TRUE(backend.saw_expected_database());
}

}  // namespace
}  // namespace elips::gpu
