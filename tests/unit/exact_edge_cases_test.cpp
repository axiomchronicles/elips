#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "elips/Config.hpp"
#include "elips/index_engine/ExactIndex.hpp"
#include "elips/domain/RecordID.hpp"

namespace {

using elips::ExactIndex;
using elips::Metric;
using elips::RecordID;

RecordID r() { return RecordID::generate(); }

TEST(ExactIndexEdgeTest, EmptyIndexSearchReturnsEmpty) {
    ExactIndex idx(Metric::euclidean, 3);
    auto hits = idx.search(std::vector<float>{1.0F, 0.0F, 0.0F}, 10);
    EXPECT_TRUE(hits.empty());
    EXPECT_EQ(idx.size(), 0U);
}

TEST(ExactIndexEdgeTest, RemoveNonExistentIsNoop) {
    ExactIndex idx(Metric::cosine, 2);
    RecordID fake = RecordID::generate();
    EXPECT_NO_THROW(idx.remove(fake));
    EXPECT_EQ(idx.size(), 0U);
}

TEST(ExactIndexEdgeTest, ReInsertSameId) {
    ExactIndex idx(Metric::euclidean, 1);
    RecordID id = r();
    idx.insert(id, std::vector<float>{1.0F});
    idx.insert(id, std::vector<float>{2.0F});  // re-insert
    EXPECT_EQ(idx.size(), 2U);  // appended
}

TEST(ExactIndexEdgeTest, SearchAsksZeroResults) {
    ExactIndex idx(Metric::euclidean, 2);
    idx.insert(r(), std::vector<float>{0.0F, 0.0F});
    auto hits = idx.search(std::vector<float>{1.0F, 0.0F}, 0);
    EXPECT_TRUE(hits.empty());
}

TEST(ExactIndexEdgeTest, SearchAsksMoreThanInserted) {
    ExactIndex idx(Metric::euclidean, 2);
    for (int i = 0; i < 3; ++i)
        idx.insert(r(), std::vector<float>{static_cast<float>(i), 0.0F});
    auto hits = idx.search(std::vector<float>{0.0F, 0.0F}, 100);
    EXPECT_EQ(hits.size(), 3U);
}

TEST(ExactIndexEdgeTest, HighDimensionalVector) {
    constexpr uint16_t dim = 1024;
    ExactIndex idx(Metric::cosine, dim);
    std::vector<float> v(dim, 1.0F / std::sqrt(static_cast<float>(dim)));
    idx.insert(r(), v);
    EXPECT_EQ(idx.size(), 1U);
    auto hits = idx.search(v, 1);
    EXPECT_EQ(hits.size(), 1U);
}

TEST(ExactIndexEdgeTest, AllMetricsWork) {
    for (auto m : {Metric::cosine, Metric::euclidean, Metric::dot_product}) {
        ExactIndex idx(m, 2);
        RecordID id = r();
        idx.insert(id, std::vector<float>{1.0F, 0.0F});
        auto hits = idx.search(std::vector<float>{1.0F, 0.0F}, 1);
        EXPECT_EQ(hits.size(), 1U);
        EXPECT_EQ(hits[0].first, id);
    }
}

TEST(ExactIndexEdgeTest, RemoveThenReinsertAtEnd) {
    ExactIndex idx(Metric::euclidean, 1);
    RecordID a = r(), b = r();
    idx.insert(a, std::vector<float>{1.0F});
    idx.insert(b, std::vector<float>{2.0F});
    idx.remove(a);
    RecordID c = r();
    idx.insert(c, std::vector<float>{3.0F});
    EXPECT_EQ(idx.size(), 2U);
    auto hits = idx.search(std::vector<float>{3.0F}, 2);
    ASSERT_EQ(hits.size(), 2U);
}

TEST(ExactIndexEdgeTest, TypeNameIsCorrectAcrossMetrics) {
    for (auto m : {Metric::cosine, Metric::euclidean, Metric::dot_product}) {
        ExactIndex idx(m, 4);
        EXPECT_EQ(idx.type_name(), "exact");
    }
}

}  // namespace
