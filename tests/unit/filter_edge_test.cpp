#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "elips/elips.hpp"
#include "elips/metadata/Filter.hpp"

namespace {

using elips::Config;

TEST(FilterEdgeTest, DefaultFilterMatchesEverything) {
    elips::Filter f;
    EXPECT_TRUE(f.matches_all());
    elips::Payload p{{"x", std::int64_t{1}}};
    EXPECT_TRUE(f.matches(p));
}

TEST(FilterEdgeTest, EqualityComparisonTypes) {
    auto match_int = elips::Filter().field("v").equals(std::int64_t{42});
    EXPECT_TRUE(match_int.matches({{"v", std::int64_t{42}}}));
    EXPECT_FALSE(match_int.matches({{"v", std::int64_t{99}}}));

    auto match_str = elips::Filter().field("v").equals(std::string{"hello"});
    EXPECT_TRUE(match_str.matches({{"v", std::string{"hello"}}}));
    EXPECT_FALSE(match_str.matches({{"v", std::string{"world"}}}));

    auto match_bool = elips::Filter().field("v").equals(true);
    EXPECT_TRUE(match_bool.matches({{"v", true}}));
    EXPECT_FALSE(match_bool.matches({{"v", false}}));

    auto match_double = elips::Filter().field("v").equals(3.14);
    EXPECT_TRUE(match_double.matches({{"v", 3.14}}));
    EXPECT_FALSE(match_double.matches({{"v", 2.71}}));
}

TEST(FilterEdgeTest, InequalityComparisons) {
    auto gt = elips::Filter().field("v").gt(std::int64_t{10});
    EXPECT_TRUE(gt.matches({{"v", std::int64_t{11}}}));
    EXPECT_FALSE(gt.matches({{"v", std::int64_t{10}}}));
    EXPECT_FALSE(gt.matches({{"v", std::int64_t{9}}}));

    auto gte = elips::Filter().field("v").ge(std::int64_t{10});
    EXPECT_TRUE(gte.matches({{"v", std::int64_t{10}}}));
    EXPECT_TRUE(gte.matches({{"v", std::int64_t{11}}}));
    EXPECT_FALSE(gte.matches({{"v", std::int64_t{9}}}));

    auto lt = elips::Filter().field("v").lt(std::int64_t{10});
    EXPECT_TRUE(lt.matches({{"v", std::int64_t{9}}}));
    EXPECT_FALSE(lt.matches({{"v", std::int64_t{10}}}));

    auto le = elips::Filter().field("v").le(std::int64_t{10});
    EXPECT_TRUE(le.matches({{"v", std::int64_t{10}}}));
    EXPECT_TRUE(le.matches({{"v", std::int64_t{9}}}));
    EXPECT_FALSE(le.matches({{"v", std::int64_t{11}}}));
}

TEST(FilterEdgeTest, NotOperator) {
    auto f = elips::Filter::not_(elips::Filter().field("x").equals(true));
    EXPECT_FALSE(f.matches({{"x", true}}));
    EXPECT_TRUE(f.matches({{"x", false}}));
    EXPECT_TRUE(f.matches({{"y", std::int64_t{1}}}));  // missing field
}

TEST(FilterEdgeTest, AndOrComposition) {
    auto a = elips::Filter().field("a").equals(std::int64_t{1});
    auto b = elips::Filter().field("b").equals(std::int64_t{2});

    EXPECT_TRUE(a.and_(b).matches({{"a", std::int64_t{1}}, {"b", std::int64_t{2}}}));
    EXPECT_FALSE(a.and_(b).matches({{"a", std::int64_t{1}}}));
    EXPECT_TRUE(a.or_(b).matches({{"a", std::int64_t{1}}}));
    EXPECT_TRUE(a.or_(b).matches({{"b", std::int64_t{2}}}));
    EXPECT_FALSE(a.or_(b).matches({{"c", std::int64_t{3}}}));
}

TEST(FilterEdgeTest, DeeplyNestedAndOr) {
    auto a = elips::Filter().field("a").equals(std::int64_t{1});
    auto b = elips::Filter().field("b").equals(std::int64_t{2});
    auto c = elips::Filter().field("c").equals(std::int64_t{3});
    auto d = elips::Filter().field("d").equals(std::int64_t{4});

    auto complex = a.and_(b).or_(c.and_(d));
    EXPECT_TRUE(complex.matches({{"a", std::int64_t{1}}, {"b", std::int64_t{2}}}));
    EXPECT_TRUE(complex.matches({{"c", std::int64_t{3}}, {"d", std::int64_t{4}}}));
    EXPECT_FALSE(complex.matches({{"a", std::int64_t{1}}}));
}

TEST(FilterEdgeTest, ContainsSubstring) {
    auto f = elips::Filter().field("title").contains("hello");
    EXPECT_TRUE(f.matches({{"title", std::string{"hello world"}}}));
    EXPECT_TRUE(f.matches({{"title", std::string{"hello"}}}));
    EXPECT_FALSE(f.matches({{"title", std::string{"world"}}}));
    EXPECT_FALSE(f.matches({{"title", std::string{""}}}));
    EXPECT_FALSE(f.matches({{"other", std::string{"hello"}}}));
}

TEST(FilterEdgeTest, OneOfSetMembership) {
    auto f = elips::Filter().field("tag").one_of(
        {std::string{"a"}, std::string{"b"}, std::string{"c"}});
    EXPECT_TRUE(f.matches({{"tag", std::string{"a"}}}));
    EXPECT_TRUE(f.matches({{"tag", std::string{"c"}}}));
    EXPECT_FALSE(f.matches({{"tag", std::string{"d"}}}));
    EXPECT_FALSE(f.matches({{"other", std::string{"a"}}}));
}

TEST(FilterEdgeTest, EmptyOneOfNeverMatches) {
    auto f = elips::Filter().field("x").one_of({});
    EXPECT_FALSE(f.matches({{"x", std::int64_t{1}}}));
}

TEST(FilterEdgeTest, MissingFieldDoesNotMatch) {
    auto f = elips::Filter().field("missing").equals(std::int64_t{1});
    EXPECT_FALSE(f.matches({{"present", std::int64_t{1}}}));
}

TEST(FilterEdgeTest, ChainedBuilderProducesAnd) {
    auto f = elips::Filter()
        .field("a").equals(std::int64_t{1})
        .field("b").gt(std::int64_t{0});

    EXPECT_TRUE(f.matches({{"a", std::int64_t{1}}, {"b", std::int64_t{5}}}));
    EXPECT_FALSE(f.matches({{"a", std::int64_t{1}}, {"b", std::int64_t{0}}}));
    EXPECT_FALSE(f.matches({{"a", std::int64_t{2}}, {"b", std::int64_t{5}}}));
}

TEST(FilterEdgeTest, NotEqualOperator) {
    auto f = elips::Filter().field("x").not_equals(std::int64_t{0});
    EXPECT_TRUE(f.matches({{"x", std::int64_t{1}}}));
    EXPECT_FALSE(f.matches({{"x", std::int64_t{0}}}));
}

}  // namespace