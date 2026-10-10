#include <gtest/gtest.h>

#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/gpu_engine/BackendSelector.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuBruteForceIndex.hpp"
#include "elips/gpu_engine/GpuGraphIndex.hpp"
#include "elips/gpu_engine/GpuSearchPipeline.hpp"
#include "elips/domain/Vector.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/IndexFlat.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu {
namespace {

class GpuBackendIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        GpuDeviceManager manager;
        auto devices = manager.probe_all_devices();
        BackendSelector selector;
        GpuConfig config;
        config.policy = GpuPolicy::Auto;
        auto sel = selector.select(config, devices);
        if (sel.has_value()) {
            backend_ = std::move(*sel);
        }
    }

    std::unique_ptr<GpuBackend> backend_;
};

TEST_F(GpuBackendIntegrationTest, initialize_succeeds) {
    if (!backend_) GTEST_SKIP() << "No GPU available";
    EXPECT_TRUE(backend_->is_available());
}

TEST(GpuBackendLifecycleTest, repeated_selection_reports_non_empty_device_metadata) {
    GpuDeviceManager manager;
    BackendSelector selector;
    GpuConfig config;
    config.policy = GpuPolicy::Auto;

    constexpr int kIterations = 25;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        auto devices = manager.probe_all_devices();
        auto selected = selector.select(config, devices);
        if (!selected.has_value()) {
            GTEST_SKIP() << "No GPU available";
        }

        const auto info = (*selected)->device_info();
        EXPECT_FALSE(info.name.empty()) << "iteration " << iteration;
        EXPECT_FALSE(info.vendor.empty()) << "iteration " << iteration;
        EXPECT_FALSE(info.backend.empty()) << "iteration " << iteration;
    }
}

TEST_F(GpuBackendIntegrationTest, allocate_upload_download_roundtrip) {
    if (!backend_) GTEST_SKIP() << "No GPU available";

    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    size_t bytes = host_data.size() * sizeof(float);

    auto buf = backend_->allocate_device(bytes);
    ASSERT_TRUE(buf.has_value());

    auto up = backend_->upload(host_data.data(), *buf, bytes);
    ASSERT_TRUE(up.has_value());

    std::vector<float> host_out(8, 0.0f);
    auto down = backend_->download(*buf, host_out.data(), bytes);
    ASSERT_TRUE(down.has_value());

    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_data[i], host_out[i]);
    }

    backend_->free_device(std::move(*buf));
}

TEST_F(GpuBackendIntegrationTest,
       repeated_allocation_transfer_cleanup_remains_stable) {
    if (!backend_) GTEST_SKIP() << "No GPU available";

    constexpr size_t kIterations = 64;
    constexpr size_t kElementCount = 128;
    const size_t bytes = kElementCount * sizeof(float);

    std::vector<float> host_data(kElementCount);
    std::vector<float> host_out(kElementCount);
    for (size_t iteration = 0; iteration < kIterations; ++iteration) {
        for (size_t i = 0; i < kElementCount; ++i) {
            host_data[i] =
                static_cast<float>((iteration + i) % 17) / 16.0f;
        }

        auto buf = backend_->allocate_device(bytes);
        ASSERT_TRUE(buf.has_value()) << "iteration " << iteration;

        auto up = backend_->upload(host_data.data(), *buf, bytes);
        ASSERT_TRUE(up.has_value()) << "iteration " << iteration;

        std::fill(host_out.begin(), host_out.end(), 0.0f);
        auto down = backend_->download(*buf, host_out.data(), bytes);
        ASSERT_TRUE(down.has_value()) << "iteration " << iteration;

        for (size_t i = 0; i < host_data.size(); ++i) {
            EXPECT_FLOAT_EQ(host_data[i], host_out[i])
                << "iteration " << iteration << " element " << i;
        }

        backend_->free_device(std::move(*buf));
    }

    backend_->synchronize();
    EXPECT_TRUE(backend_->is_idle());
}

TEST_F(GpuBackendIntegrationTest, cosine_distance_matches_cpu) {
    if (!backend_) GTEST_SKIP() << "No GPU available";

    std::vector<float> db_data = {
        1.0f, 0.0f, 0.0f,  // normalized
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    std::vector<float> query = {1.0f, 0.0f, 0.0f};  // normalized
    size_t nq = 1, nb = 3, dim = 3;

    auto db_buf = backend_->allocate_device(db_data.size() * sizeof(float));
    auto q_buf = backend_->allocate_device(query.size() * sizeof(float));
    ASSERT_TRUE(db_buf.has_value());
    ASSERT_TRUE(q_buf.has_value());

    backend_->upload(db_data.data(), *db_buf, db_data.size() * sizeof(float));
    backend_->upload(query.data(), *q_buf, query.size() * sizeof(float));

    std::vector<float> dists(nq * nb);

    auto* db_ptr = static_cast<const float*>(db_buf->device_ptr());
    auto* q_ptr = static_cast<const float*>(q_buf->device_ptr());

    auto result = backend_->compute_distances_batch(
        std::span<const float>{q_ptr, dim},
        std::span<const float>{db_ptr, nb * dim},
        dists, nq, nb, dim, Metric::cosine);
    ASSERT_TRUE(result.has_value());

    float cpu_dist0 = distance(Metric::cosine,
        std::span<const float>{query.data(), dim},
        std::span<const float>{db_data.data(), dim});
    EXPECT_NEAR(dists[0], cpu_dist0, 1e-4f);

    backend_->free_device(std::move(*db_buf));
    backend_->free_device(std::move(*q_buf));
}

TEST_F(GpuBackendIntegrationTest, brute_force_index_search) {
    if (!backend_) GTEST_SKIP() << "No GPU available";

    GpuConfig gconfig;
    gconfig.algorithm = GpuIndexAlgorithm::BruteForce;

    GpuBruteForceIndex index(*backend_, Metric::cosine, 3, gconfig);

    RecordID id0 = RecordID::generate();
    RecordID id1 = RecordID::generate();
    RecordID id2 = RecordID::generate();

    std::vector<float> vec0 = {1.0f, 0.0f, 0.0f};
    std::vector<float> vec1 = {0.0f, 1.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 0.0f, 1.0f};

    std::vector<float> all_vectors = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    std::vector<RecordID> ids = {id0, id1, id2};

    auto build = index.build_from_batch(all_vectors, ids, GpuIndexBuildParams{});
    ASSERT_TRUE(build.has_value());

    EXPECT_EQ(index.ntotal(), 3u);

    std::vector<float> query = {1.0f, 0.0f, 0.0f};
    auto results = index.search(query, 3);
    EXPECT_EQ(results.size(), 3u);
}

TEST_F(GpuBackendIntegrationTest, brute_force_index_replace_same_id_is_stable) {
    if (!backend_) GTEST_SKIP() << "No GPU available";

    GpuConfig gconfig;
    gconfig.algorithm = GpuIndexAlgorithm::BruteForce;

    GpuBruteForceIndex index(*backend_, Metric::cosine, 2, gconfig);

    const RecordID id = RecordID::generate();

    index.add_with_ids(id, std::vector<float>{1.0f, 0.0f});
    index.remove(id);
    index.add_with_ids(id, std::vector<float>{0.0f, 1.0f});

    const auto results = index.search(std::vector<float>{0.0f, 1.0f}, 1);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].first, id);
}

}  // namespace
}  // namespace elips::gpu
