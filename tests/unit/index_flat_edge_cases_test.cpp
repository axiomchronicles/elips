#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "elips/Config.hpp"
#include "elips/index_engine/IndexFlat.hpp"
#include "elips/domain/RecordID.hpp"

namespace {

using elips::IndexFlat;
using elips::Metric;
using elips::RecordID;

RecordID r() { return RecordID::generate(); }

TEST(IndexFlatEdgeTest, EmptyIndexSearchReturnsEmpty) {
    IndexFlat idx(Metric::euclidean, 3);
    auto hits = idx.search(std::vector<float>{1.0F, 0.0F, 0.0F}, 10);
    EXPECT_TRUE(hits.empty());
    EXPECT_EQ(idx.ntotal(), 0U);
}

TEST(IndexFlatEdgeTest, RemoveNonExistentIsNoop) {
    IndexFlat idx(Metric::cosine, 2);
    RecordID fake = RecordID::generate();
    EXPECT_NO_THROW(idx.remove(fake));
    EXPECT_EQ(idx.ntotal(), 0U);
}

TEST(IndexFlatEdgeTest, ReInsertSameId) {
    IndexFlat idx(Metric::euclidean, 1);
    RecordID id = r();
    idx.add_with_ids(id, std::vector<float>{1.0F});
    idx.add_with_ids(id, std::vector<float>{2.0F});  // re-insert
    EXPECT_EQ(idx.ntotal(), 2U);  // appended
}

TEST(IndexFlatEdgeTest, SearchAsksZeroResults) {
    IndexFlat idx(Metric::euclidean, 2);
    idx.add_with_ids(r(), std::vector<float>{0.0F, 0.0F});
    auto hits = idx.search(std::vector<float>{1.0F, 0.0F}, 0);
    EXPECT_TRUE(hits.empty());
}

TEST(IndexFlatEdgeTest, SearchAsksMoreThanInserted) {
    IndexFlat idx(Metric::euclidean, 2);
    for (int i = 0; i < 3; ++i)
        idx.add_with_ids(r(), std::vector<float>{static_cast<float>(i), 0.0F});
    auto hits = idx.search(std::vector<float>{0.0F, 0.0F}, 100);
    EXPECT_EQ(hits.size(), 3U);
}

TEST(IndexFlatEdgeTest, HighDimensionalVector) {
    constexpr uint16_t dim = 1024;
    IndexFlat idx(Metric::cosine, dim);
    std::vector<float> v(dim, 1.0F / std::sqrt(static_cast<float>(dim)));
    idx.add_with_ids(r(), v);
    EXPECT_EQ(idx.ntotal(), 1U);
    auto hits = idx.search(v, 1);
    EXPECT_EQ(hits.size(), 1U);
}

TEST(IndexFlatEdgeTest, AllMetricsWork) {
    for (auto m : {Metric::cosine, Metric::euclidean, Metric::dot_product}) {
        IndexFlat idx(m, 2);
        RecordID id = r();
        idx.add_with_ids(id, std::vector<float>{1.0F, 0.0F});
        auto hits = idx.search(std::vector<float>{1.0F, 0.0F}, 1);
        EXPECT_EQ(hits.size(), 1U);
        EXPECT_EQ(hits[0].first, id);
    }
}

TEST(IndexFlatEdgeTest, RemoveThenReinsertAtEnd) {
    IndexFlat idx(Metric::euclidean, 1);
    RecordID a = r(), b = r();
    idx.add_with_ids(a, std::vector<float>{1.0F});
    idx.add_with_ids(b, std::vector<float>{2.0F});
    idx.remove(a);
    RecordID c = r();
    idx.add_with_ids(c, std::vector<float>{3.0F});
    EXPECT_EQ(idx.ntotal(), 2U);
    auto hits = idx.search(std::vector<float>{3.0F}, 2);
    ASSERT_EQ(hits.size(), 2U);
}

TEST(IndexFlatEdgeTest, TypeNameIsCorrectAcrossMetrics) {
    for (auto m : {Metric::cosine, Metric::euclidean, Metric::dot_product}) {
        IndexFlat idx(m, 4);
        EXPECT_EQ(idx.type_name(), "exact");
    }
}

}  // namespace
