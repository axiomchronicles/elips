#include <gtest/gtest.h>

#include <vector>

#include "elips/Config.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace {

using elips::distance;
using elips::Metric;

TEST(MetricsTest, CosineIdenticalNormalizedVectorsZeroDistance) {
    const std::vector<float> a{1.0F, 0.0F};  // already unit length
    EXPECT_NEAR(distance(Metric::cosine, a, a), 0.0F, 1e-6F);
}

TEST(MetricsTest, CosineOrthogonalNormalizedVectorsDistanceOne) {
    const std::vector<float> a{1.0F, 0.0F};
    const std::vector<float> b{0.0F, 1.0F};
    EXPECT_NEAR(distance(Metric::cosine, a, b), 1.0F, 1e-6F);
}

TEST(MetricsTest, EuclideanMatchesManualComputation) {
    const std::vector<float> a{0.0F, 0.0F};
    const std::vector<float> b{3.0F, 4.0F};
    EXPECT_NEAR(distance(Metric::euclidean, a, b), 5.0F, 1e-6F);
}

TEST(MetricsTest, DotProductIsNegatedForAscendingSort) {
    const std::vector<float> a{1.0F, 2.0F, 3.0F};
    const std::vector<float> b{1.0F, 1.0F, 1.0F};
    // dot = 6, distance = -6 so larger dot products sort first.
    EXPECT_NEAR(distance(Metric::dot_product, a, b), -6.0F, 1e-6F);
}

TEST(MetricsTest, OnlyCosineRequiresNormalization) {
    EXPECT_TRUE(elips::requires_normalization(Metric::cosine));
    EXPECT_FALSE(elips::requires_normalization(Metric::euclidean));
    EXPECT_FALSE(elips::requires_normalization(Metric::dot_product));
}

}  // namespace
