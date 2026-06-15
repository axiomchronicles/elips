#include <gtest/gtest.h>

#include <random>
#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/gpu_engine/GpuSelector.hpp"
#include "elips/gpu_engine/GpuBruteForceIndex.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/index_engine/ExactIndex.hpp"
#include "elips/domain/Vector.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu {
namespace {

float compute_recall(const std::vector<IndexPort::Hit>& ground,
                     const std::vector<IndexPort::Hit>& actual,
                     size_t k) {
    size_t matches = 0;
    for (size_t i = 0; i < std::min(k, actual.size()); ++i) {
        for (size_t j = 0; j < std::min(k, ground.size()); ++j) {
            if (actual[i].first == ground[j].first) {
                ++matches;
                break;
            }
        }
    }
    return static_cast<float>(matches) / static_cast<float>(k);
}

class CpuGpuRecallParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        GpuDeviceManager manager;
        auto devices = manager.probe_all_devices();
        GpuSelector selector;
        GpuConfig config;
        config.policy = GpuPolicy::Auto;
        auto sel = selector.select(config, devices);
        if (sel.has_value()) {
            backend_ = std::move(*sel);
        }
        cpu_index_ = std::make_unique<ExactIndex>(Metric::cosine, 128);
    }

    std::unique_ptr<GpuPort> backend_;
    std::unique_ptr<ExactIndex> cpu_index_;
};

TEST_F(CpuGpuRecallParityTest, brute_force_recall_perfect) {
    if (!backend_) GTEST_SKIP() << "No GPU available";

    const size_t n = 100;
    const size_t dim = 8;
    const size_t k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(n * dim);
    for (auto& v : vectors) v = dist(rng);

    std::vector<RecordID> ids;
    for (size_t i = 0; i < n; ++i) ids.push_back(RecordID::generate());

    for (size_t i = 0; i < n; ++i) {
        cpu_index_->insert(ids[i], {vectors.data() + i * dim, dim});
    }

    GpuConfig gconfig;
    gconfig.algorithm = GpuIndexAlgorithm::BruteForce;
    GpuBruteForceIndex gpu_idx(*backend_, Metric::cosine,
                               static_cast<uint16_t>(dim), gconfig);
    auto build = gpu_idx.build_from_batch(vectors, ids, GpuIndexBuildParams{});
    ASSERT_TRUE(build.has_value());

    std::vector<float> query(dim);
    for (auto& v : query) v = dist(rng);

    auto ground = cpu_index_->search(query, k);
    auto actual = gpu_idx.search(query, k);

    EXPECT_EQ(ground.size(), k);
    EXPECT_EQ(actual.size(), k);

    float recall = compute_recall(ground, actual, k);
    EXPECT_GE(recall, 0.95f);
}

}  // namespace
}  // namespace elips::gpu