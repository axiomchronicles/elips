#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "elips/domain/Vector.hpp"

namespace {

using elips::Vector;

TEST(VectorTest, DefaultConstructedIsEmpty) {
    Vector v;
    EXPECT_TRUE(v.empty());
    EXPECT_EQ(v.dimension(), 0U);
    EXPECT_EQ(v.values().size(), 0U);
}

TEST(VectorTest, ConstructedWithValues) {
    Vector v{{1.0F, 2.0F, 3.0F}};
    EXPECT_FALSE(v.empty());
    EXPECT_EQ(v.dimension(), 3U);
    EXPECT_EQ(v.values()[0], 1.0F);
    EXPECT_EQ(v.values()[1], 2.0F);
    EXPECT_EQ(v.values()[2], 3.0F);
}

TEST(VectorTest, MagnitudeComputation) {
    Vector v{{3.0F, 4.0F}};
    EXPECT_FLOAT_EQ(v.magnitude(), 5.0F);
}

TEST(VectorTest, ZeroVectorMagnitude) {
    Vector v{{0.0F, 0.0F, 0.0F}};
    EXPECT_FLOAT_EQ(v.magnitude(), 0.0F);
}

TEST(VectorTest, NormalizedIsUnitLength) {
    Vector v{{3.0F, 4.0F}};
    Vector n = v.normalized();
    EXPECT_FLOAT_EQ(n.magnitude(), 1.0F);
}

TEST(VectorTest, NormalizedZeroVectorIsUnchanged) {
    Vector v{{0.0F, 0.0F}};
    Vector n = v.normalized();
    EXPECT_FLOAT_EQ(n.values()[0], 0.0F);
    EXPECT_FLOAT_EQ(n.values()[1], 0.0F);
}

TEST(VectorTest, SingleElementNormalization) {
    Vector v{{7.0F}};
    Vector n = v.normalized();
    EXPECT_FLOAT_EQ(n.magnitude(), 1.0F);
    EXPECT_FLOAT_EQ(n.values()[0], 1.0F);
}

TEST(VectorTest, LargeVectorNormalization) {
    constexpr uint16_t dim = 1024;
    std::vector<float> data(dim, 1.0F);
    Vector v{std::move(data)};
    Vector n = v.normalized();
    EXPECT_NEAR(n.magnitude(), 1.0F, 1e-4F);
}

TEST(VectorTest, NegativeValuesNormalization) {
    Vector v{{-3.0F, -4.0F}};
    Vector n = v.normalized();
    EXPECT_FLOAT_EQ(n.magnitude(), 1.0F);
    EXPECT_FLOAT_EQ(n.values()[0], -0.6F);
    EXPECT_FLOAT_EQ(n.values()[1], -0.8F);
}

TEST(VectorTest, EmptyVectorNormalization) {
    Vector v;
    Vector n = v.normalized();
    EXPECT_TRUE(n.empty());
}

TEST(VectorTest, DimensionMatchesSize) {
    for (uint16_t dim : {0U, 1U, 16U, 128U, 1024U}) {
        std::vector<float> data(dim, 0.0F);
        Vector v{std::move(data)};
        EXPECT_EQ(v.dimension(), dim);
    }
}

}  // namespace