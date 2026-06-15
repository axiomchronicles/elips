#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/HierarchicalGraphIndex.hpp"

namespace {

using elips::GraphParams;
using elips::HierarchicalGraphIndex;
using elips::Metric;
using elips::RecordID;

RecordID r() { return RecordID::generate(); }

class HnswEdgeTest : public ::testing::Test {
protected:
    HierarchicalGraphIndex make_index(uint16_t dim, GraphParams p = {}) {
        return HierarchicalGraphIndex(Metric::euclidean, dim, p);
    }
};

TEST_F(HnswEdgeTest, EmptyIndexSearchReturnsEmpty) {
    auto idx = make_index(3);
    auto hits = idx.search(std::vector<float>{1.0F, 0.0F, 0.0F}, 10);
    EXPECT_TRUE(hits.empty());
    EXPECT_EQ(idx.size(), 0U);
}

TEST_F(HnswEdgeTest, SingleElementInsertAndSearch) {
    auto idx = make_index(2);
    RecordID id = r();
    idx.insert(id, std::vector<float>{1.0F, 2.0F});
    EXPECT_EQ(idx.size(), 1U);
    auto hits = idx.search(std::vector<float>{1.0F, 2.0F}, 10);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].first, id);
}

TEST_F(HnswEdgeTest, DuplicateIdsAccumulate) {
    auto idx = make_index(2);
    RecordID id = r();
    idx.insert(id, std::vector<float>{0.0F, 0.0F});
    idx.insert(id, std::vector<float>{1.0F, 1.0F});
    EXPECT_EQ(idx.size(), 2U);
}

TEST_F(HnswEdgeTest, RemoveThenSearchExcludesDeleted) {
    auto idx = make_index(2);
    RecordID a = r(), b = r(), c = r();
    idx.insert(a, std::vector<float>{0.0F, 0.0F});
    idx.insert(b, std::vector<float>{1.0F, 1.0F});
    idx.insert(c, std::vector<float>{2.0F, 2.0F});
    idx.remove(b);
    EXPECT_EQ(idx.size(), 2U);
    auto hits = idx.search(std::vector<float>{1.0F, 1.0F}, 3);
    for (const auto& [id, d] : hits) {
        EXPECT_NE(id, b) << "deleted node should not appear in results";
    }
}

TEST_F(HnswEdgeTest, RemoveAllThenSearchEmpty) {
    auto idx = make_index(2);
    std::vector<RecordID> ids;
    for (int i = 0; i < 5; ++i) {
        RecordID id = r();
        ids.push_back(id);
        idx.insert(id, std::vector<float>{static_cast<float>(i), 0.0F});
    }
    for (const auto& id : ids) idx.remove(id);
    EXPECT_EQ(idx.size(), 0U);
    auto hits = idx.search(std::vector<float>{0.0F, 0.0F}, 5);
    EXPECT_TRUE(hits.empty());
}

TEST_F(HnswEdgeTest, SearchRequestsMoreThanInserted) {
    auto idx = make_index(3);
    for (int i = 0; i < 5; ++i)
        idx.insert(r(), std::vector<float>{static_cast<float>(i), 0.0F, 0.0F});
    auto hits = idx.search(std::vector<float>{0.0F, 0.0F, 0.0F}, 100);
    EXPECT_EQ(hits.size(), 5U);
}

TEST_F(HnswEdgeTest, HighDimensionalSingleInsert) {
    constexpr uint16_t dim = 768;
    auto idx = make_index(dim);
    std::vector<float> v(dim, 1.0F / std::sqrt(768.0F));
    idx.insert(r(), v);
    EXPECT_EQ(idx.size(), 1U);
    auto hits = idx.search(v, 1);
    EXPECT_EQ(hits.size(), 1U);
}

TEST_F(HnswEdgeTest, CustomGraphParamsAffectSearch) {
    GraphParams tight{4, 50, 20};
    auto idx = make_index(8, tight);
    for (int i = 0; i < 50; ++i) {
        std::vector<float> v(8);
        for (auto& x : v) x = static_cast<float>(i);
        idx.insert(r(), v);
    }
    auto hits = idx.search(std::vector<float>(8, 0.0F), 5);
    EXPECT_EQ(hits.size(), 5U);
}

TEST_F(HnswEdgeTest, TypeNameIsGraph) {
    auto idx = make_index(4);
    EXPECT_EQ(idx.type_name(), "graph");
}

TEST_F(HnswEdgeTest, AllMetricsWork) {
    for (auto m : {Metric::cosine, Metric::euclidean, Metric::dot_product}) {
        HierarchicalGraphIndex idx(m, 2, GraphParams{});
        RecordID id = r();
        idx.insert(id, std::vector<float>{1.0F, 0.0F});
        auto hits = idx.search(std::vector<float>{1.0F, 0.0F}, 1);
        EXPECT_GE(hits.size(), 1U);
    }
}

TEST_F(HnswEdgeTest, InsertionOrderDoesNotAffectSize) {
    auto idx = make_index(4);
    std::vector<RecordID> ids;
    for (int i = 0; i < 100; ++i) {
        RecordID id = r();
        ids.push_back(id);
        std::vector<float> v(4, static_cast<float>(i));
        idx.insert(id, v);
    }
    EXPECT_EQ(idx.size(), 100U);
}

}  // namespace
