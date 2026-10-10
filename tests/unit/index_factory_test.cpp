#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/Errors.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/Index.hpp"
#include "elips/index_engine/IndexFactory.hpp"

namespace {

using elips::index_factory;
using elips::Metric;
using elips::RecordID;

TEST(IndexFactoryTest, FlatSpec) {
    auto idx = index_factory(4, Metric::euclidean, "Flat");
    EXPECT_EQ(idx->type_name(), "exact");
    idx->add_with_ids(RecordID::generate(), std::vector<float>{0, 0, 0, 0});
    idx->add_with_ids(RecordID::generate(), std::vector<float>{1, 1, 1, 1});
    EXPECT_EQ(idx->ntotal(), 2U);
    EXPECT_EQ(idx->search(std::vector<float>{0, 0, 0, 0}, 1).size(), 1U);
}

TEST(IndexFactoryTest, IvfFlatSpec) {
    auto idx = index_factory(2, Metric::euclidean, "IVF64,Flat");
    EXPECT_EQ(idx->type_name(), "ivfflat");
    EXPECT_FALSE(idx->is_trained());

    std::vector<float> train;
    for (int i = 0; i < 128; ++i) {
        train.push_back(static_cast<float>(i % 7));
        train.push_back(static_cast<float>(i % 11));
    }
    idx->train(train, 128);
    EXPECT_TRUE(idx->is_trained());
    idx->add_with_ids(RecordID::generate(), std::vector<float>{0, 0});
    idx->add_with_ids(RecordID::generate(), std::vector<float>{5, 5});
    EXPECT_EQ(idx->ntotal(), 2U);
    EXPECT_FALSE(idx->search(std::vector<float>{0, 0}, 2).empty());
}

TEST(IndexFactoryTest, HnswSpec) {
    auto idx = index_factory(8, Metric::euclidean, "HNSW32");
    EXPECT_EQ(idx->type_name(), "graph");
    for (int i = 0; i < 10; ++i) {
        std::vector<float> v(8, static_cast<float>(i));
        idx->add_with_ids(RecordID::generate(), v);
    }
    EXPECT_EQ(idx->ntotal(), 10U);
    EXPECT_EQ(idx->search(std::vector<float>(8, 0.0F), 3).size(), 3U);
}

TEST(IndexFactoryTest, UnknownSpecThrows) {
    EXPECT_THROW((void)index_factory(4, Metric::euclidean, "Bogus"),
                 elips::ConfigError);
    EXPECT_THROW((void)index_factory(4, Metric::euclidean, "IVF,Flat"),
                 elips::ConfigError);
    EXPECT_THROW((void)index_factory(4, Metric::euclidean, "IVF64,PQ8"),
                 elips::ConfigError);
    EXPECT_THROW((void)index_factory(4, Metric::euclidean, "HNSWx"),
                 elips::ConfigError);
}

}  // namespace
