#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/ExactIndex.hpp"
#include "elips/index_engine/HierarchicalGraphIndex.hpp"

namespace {

using elips::ExactIndex;
using elips::GraphParams;
using elips::HierarchicalGraphIndex;
using elips::Metric;
using elips::RecordID;

// HNSW should recover the vast majority of true nearest neighbors that the
// brute-force ExactIndex (the ground-truth oracle) returns.
TEST(HnswRecallTest, RecallAtTenIsHighVsExactGroundTruth) {
    constexpr std::size_t count = 2000;
    constexpr std::uint16_t dim = 64;
    constexpr std::size_t k = 10;
    constexpr std::size_t queries = 100;

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0F, 1.0F);

    HierarchicalGraphIndex graph(Metric::euclidean, dim, GraphParams{});
    ExactIndex exact(Metric::euclidean, dim);

    std::vector<std::vector<float>> data(count, std::vector<float>(dim));
    for (std::size_t i = 0; i < count; ++i) {
        for (auto& v : data[i]) {
            v = dist(rng);
        }
        const RecordID id = RecordID::generate();
        graph.insert(id, data[i]);
        exact.insert(id, data[i]);
    }

    std::size_t hits = 0;
    std::size_t total = 0;
    for (std::size_t q = 0; q < queries; ++q) {
        std::vector<float> query(dim);
        for (auto& v : query) {
            v = dist(rng);
        }
        const auto truth = exact.search(query, k);
        const auto approx = graph.search(query, k);

        std::unordered_set<RecordID> truth_ids;
        for (const auto& [id, d] : truth) {
            truth_ids.insert(id);
        }
        for (const auto& [id, d] : approx) {
            if (truth_ids.count(id) != 0) {
                ++hits;
            }
        }
        total += truth.size();
    }

    const double recall = static_cast<double>(hits) / static_cast<double>(total);
    EXPECT_GE(recall, 0.90) << "recall@" << k << " = " << recall;
}

TEST(HnswRecallTest, SoftRemoveExcludesFromResults) {
    HierarchicalGraphIndex graph(Metric::euclidean, 2, GraphParams{});
    const RecordID a = RecordID::generate();
    const RecordID b = RecordID::generate();
    graph.insert(a, std::vector<float>{0.0F, 0.0F});
    graph.insert(b, std::vector<float>{5.0F, 5.0F});

    graph.remove(a);
    EXPECT_EQ(graph.size(), 1U);
    const auto hits = graph.search(std::vector<float>{0.0F, 0.0F}, 5);
    for (const auto& [id, d] : hits) {
        EXPECT_NE(id, a);
    }
}

}  // namespace
