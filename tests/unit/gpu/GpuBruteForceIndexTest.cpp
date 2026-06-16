#include <gtest/gtest.h>

#include <vector>

#include "TestRecordingBackend.hpp"
#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/gpu_engine/GpuBruteForceIndex.hpp"

namespace elips::gpu {
namespace {

TEST(GpuBruteForceIndexTest,
     search_uses_host_database_when_device_pointer_is_unreadable) {
    test::RecordingBackend backend;
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
    test::RecordingBackend backend;
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
