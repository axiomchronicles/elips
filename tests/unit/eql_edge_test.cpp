#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "elips/elips.hpp"
#include "elips/query_engine/EQLLexer.hpp"
#include "elips/query_engine/EQLParser.hpp"

namespace {

TEST(EqlEdgeTest, ParseWithComments) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    auto results = db->query(
        "# this is a comment\n"
        "seek in v nearest [1.0, 0.0] top 3 yield");
    EXPECT_TRUE(results.empty());
}

TEST(EqlEdgeTest, ParseEmptyStringThrows) {
    EXPECT_THROW((void)elips::eql::parse(""), elips::eql::ParseError);
    EXPECT_THROW((void)elips::eql::parse("   "), elips::eql::ParseError);
}

TEST(EqlEdgeTest, TokenizeProducesAtLeastEndToken) {
    auto tokens = elips::eql::tokenize("");
    EXPECT_EQ(tokens.size(), 1U);
    EXPECT_EQ(tokens[0].kind, elips::eql::TokenKind::end);
}

TEST(EqlEdgeTest, TokenizeStringWithEscapes) {
    auto tokens = elips::eql::tokenize(R"("hello \"world\"")");
    EXPECT_GE(tokens.size(), 1U);
    EXPECT_EQ(tokens[0].kind, elips::eql::TokenKind::string);
}

TEST(EqlEdgeTest, TokenizePunctuationTokens) {
    auto tokens = elips::eql::tokenize(">= <= != = < > , [ ] ( )");
    EXPECT_GE(tokens.size(), 4U);
    for (const auto& t : tokens) {
        if (t.kind != elips::eql::TokenKind::end) {
            EXPECT_EQ(t.kind, elips::eql::TokenKind::punct);
        }
    }
}

TEST(EqlEdgeTest, ComplexQueryParsesCorrectly) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(3).metric(
                                          elips::Metric::euclidean));
    db->vault("docs");
    db->query(R"(place in docs vector [1.0, 2.0, 3.0] data {"a": 1, "b": "x"})");
    db->query(R"(place in docs vector [4.0, 5.0, 6.0] data {"a": 2, "b": "y"})");

    std::map<std::string, elips::Vector> bindings{
        {"q", elips::Vector{{1.0F, 2.0F, 3.0F}}}};

    auto results = db->query(
        "seek in docs nearest $q top 10 where a >= 1 project b yield",
        bindings);
    EXPECT_EQ(results.size(), 2U);
}

TEST(EqlEdgeTest, FetchNonExistentReturnsEmpty) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    db->vault("v");
    auto results = db->query(R"(fetch from v id "00000000-0000-7000-8000-000000000000" yield)");
    EXPECT_TRUE(results.empty());
}

TEST(EqlEdgeTest, ScanWithLimitAndOffsetParses) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    db->vault("v");
    for (int i = 0; i < 10; ++i)
        db->query("place in v vector [" + std::to_string(i) + ", 0]");

    auto results = db->query("scan in v offset 2 limit 3 yield");
    EXPECT_EQ(results.size(), 3U);
}

TEST(EqlEdgeTest, ThresholdSearchParses) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2).metric(
                                          elips::Metric::euclidean));
    db->vault("v");
    db->query("place in v vector [1.0, 0.0]");
    db->query("place in v vector [5.0, 0.0]");

    auto results = db->query("seek in v nearest [1.0, 0.0] threshold 0.5 yield");
    EXPECT_EQ(results.size(), 1U);  // only the close one
}

TEST(EqlEdgeTest, ProjectFieldsReturnSubset) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    db->query(R"(place in v vector [1.0, 0.0] data {"a": 1, "b": 2, "c": 3})");
    auto results = db->query("seek in v nearest [1.0, 0.0] project a, c yield");
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].data.size(), 2U);
    EXPECT_TRUE(results[0].data.count("a") == 1);
    EXPECT_TRUE(results[0].data.count("c") == 1);
    EXPECT_TRUE(results[0].data.count("b") == 0);
}

TEST(EqlEdgeTest, MixedBooleanFilter) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    db->query(R"(place in v vector [1.0, 0.0] data {"x": 1, "y": 1})");
    db->query(R"(place in v vector [0.0, 1.0] data {"x": 1, "y": 0})");
    db->query(R"(place in v vector [1.0, 1.0] data {"x": 0, "y": 1})");
    db->query(R"(place in v vector [0.0, 0.0] data {"x": 0, "y": 0})");

    auto results = db->query(R"(scan in v where x = 1 and y = 1 yield)");
    EXPECT_EQ(results.size(), 1U);

    auto results_or = db->query(R"(scan in v where x = 1 or y = 1 yield)");
    EXPECT_EQ(results_or.size(), 3U);

    auto results_not = db->query(R"(scan in v where not x = 1 yield)");
    EXPECT_EQ(results_not.size(), 2U);
}

TEST(EqlEdgeTest, EmptyBindingsIsSafe) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    db->vault("v");
    auto results = db->query("seek in v nearest [1.0, 0.0] yield");
    EXPECT_TRUE(results.empty());
}

}  // namespace