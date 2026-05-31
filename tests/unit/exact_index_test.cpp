#include <gtest/gtest.h>

#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/ExactIndex.hpp"

namespace {

using elips::ExactIndex;
using elips::Metric;
using elips::RecordID;

RecordID make_id() { return RecordID::generate(); }

TEST(ExactIndexTest, ReturnsNearestFirstAscending) {
    ExactIndex index(Metric::euclidean, 2);
    const RecordID near = make_id();
    const RecordID far = make_id();
    index.insert(near, std::vector<float>{0.0F, 0.0F});
    index.insert(far, std::vector<float>{10.0F, 10.0F});

    const std::vector<float> query{0.1F, 0.1F};
    const auto hits = index.search(query, 2);

    ASSERT_EQ(hits.size(), 2U);
    EXPECT_EQ(hits[0].first, near);
    EXPECT_EQ(hits[1].first, far);
    EXPECT_LT(hits[0].second, hits[1].second);
}

TEST(ExactIndexTest, TopKCapsResultCount) {
    ExactIndex index(Metric::euclidean, 1);
    for (int i = 0; i < 5; ++i) {
        index.insert(make_id(), std::vector<float>{static_cast<float>(i)});
    }
    EXPECT_EQ(index.size(), 5U);
    EXPECT_EQ(index.search(std::vector<float>{0.0F}, 3).size(), 3U);
}

TEST(ExactIndexTest, RemoveDropsRecordFromResults) {
    ExactIndex index(Metric::euclidean, 1);
    const RecordID a = make_id();
    const RecordID b = make_id();
    index.insert(a, std::vector<float>{1.0F});
    index.insert(b, std::vector<float>{2.0F});

    index.remove(a);
    const auto hits = index.search(std::vector<float>{1.0F}, 10);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].first, b);
}

TEST(ExactIndexTest, TypeNameIsExact) {
    ExactIndex index(Metric::cosine, 4);
    EXPECT_EQ(index.type_name(), "exact");
}

}  // namespace
