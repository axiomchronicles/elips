#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "elips/Config.hpp"

namespace {

using elips::Config;
using elips::Durability;
using elips::GraphParams;
using elips::IndexType;
using elips::Metric;

TEST(ConfigTest, DefaultValuesAreSane) {
    Config c;
    EXPECT_EQ(c.dimension(), 0U);
    EXPECT_EQ(c.metric(), Metric::cosine);
    EXPECT_EQ(c.index(), IndexType::graph);
    EXPECT_EQ(c.durability(), Durability::standard);
}

TEST(ConfigTest, FluentBuilderChainSetsAllValues) {
    Config c;
    c.dimension(768).metric(Metric::euclidean).index(IndexType::exact)
        .durability(Durability::paranoid);

    EXPECT_EQ(c.dimension(), 768U);
    EXPECT_EQ(c.metric(), Metric::euclidean);
    EXPECT_EQ(c.index(), IndexType::exact);
    EXPECT_EQ(c.durability(), Durability::paranoid);
}

TEST(ConfigTest, GraphParamsDefaults) {
    Config c;
    const auto& gp = c.graph_params();
    EXPECT_EQ(gp.max_connections, 16U);
    EXPECT_EQ(gp.ef_construction, 200U);
    EXPECT_EQ(gp.ef_search, 50U);
}

TEST(ConfigTest, CustomGraphParams) {
    GraphParams gp{32, 400, 100};
    Config c;
    c.graph_params(gp);

    EXPECT_EQ(c.graph_params().max_connections, 32U);
    EXPECT_EQ(c.graph_params().ef_construction, 400U);
    EXPECT_EQ(c.graph_params().ef_search, 100U);
}

TEST(ConfigTest, AllDurabilityLevels) {
    for (auto level : {Durability::paranoid, Durability::standard,
                       Durability::relaxed, Durability::ephemeral}) {
        Config c;
        c.durability(level);
        EXPECT_EQ(c.durability(), level);
    }
}

TEST(ConfigTest, IndexTypeValues) {
    Config c;
    c.index(IndexType::graph);
    EXPECT_EQ(c.index(), IndexType::graph);

    c.index(IndexType::exact);
    EXPECT_EQ(c.index(), IndexType::exact);
}

TEST(ConfigTest, MetricToFromStringRoundTrip) {
    for (auto m : {Metric::cosine, Metric::euclidean, Metric::dot_product}) {
        std::string s{elips::to_string(m)};
        EXPECT_EQ(elips::metric_from_string(s), m);
    }
}

TEST(ConfigTest, MetricFromStringCaseSensitive) {
    EXPECT_EQ(elips::metric_from_string("cosine"), Metric::cosine);
    EXPECT_EQ(elips::metric_from_string("euclidean"), Metric::euclidean);
    EXPECT_EQ(elips::metric_from_string("dot_product"), Metric::dot_product);
}

TEST(ConfigTest, ToStringValues) {
    EXPECT_EQ(std::string(elips::to_string(Metric::cosine)), "cosine");
    EXPECT_EQ(std::string(elips::to_string(Metric::euclidean)), "euclidean");
    EXPECT_EQ(std::string(elips::to_string(Metric::dot_product)), "dot_product");
}

TEST(ConfigTest, GraphParamsConstructor) {
    GraphParams p;  // default
    EXPECT_EQ(p.max_connections, 16U);
    EXPECT_EQ(p.ef_construction, 200U);
    EXPECT_EQ(p.ef_search, 50U);

    GraphParams custom{64, 500, 200};
    EXPECT_EQ(custom.max_connections, 64U);
    EXPECT_EQ(custom.ef_construction, 500U);
    EXPECT_EQ(custom.ef_search, 200U);
}

TEST(ConfigTest, ConfigIsCopyable) {
    Config a;
    a.dimension(128).metric(Metric::euclidean);
    Config b = a;
    EXPECT_EQ(b.dimension(), 128U);
    EXPECT_EQ(b.metric(), Metric::euclidean);
}

}  // namespace