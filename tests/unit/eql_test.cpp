#include <gtest/gtest.h>

#include <map>
#include <string>

#include "elips/elips.hpp"
#include "elips/query_engine/EQLParser.hpp"

namespace {

TEST(EqlTest, ParsesSearchStatement) {
    const auto stmt = elips::eql::parse(
        "seek in docs nearest $q top 5 where year >= 2020 project title yield");
    const auto& s = std::get<elips::eql::SearchStatement>(stmt);
    EXPECT_EQ(s.vault, "docs");
    EXPECT_EQ(s.query.binding, "q");
    ASSERT_TRUE(s.top.has_value());
    EXPECT_EQ(*s.top, 5);
    ASSERT_EQ(s.projection.size(), 1U);
    EXPECT_EQ(s.projection[0], "title");
}

TEST(EqlTest, PlaceThenSeekViaEql) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(3).metric(
                                          elips::Metric::euclidean));
    (void)db->query(R"(place in mem vector [1.0, 0.0, 0.0] data {"tag": "a"})");
    (void)db->query(R"(place in mem vector [0.0, 1.0, 0.0] data {"tag": "b"})");

    const std::map<std::string, elips::Vector> bindings{
        {"q", elips::Vector{{0.9F, 0.1F, 0.0F}}}};
    const auto results =
        db->query("seek in mem nearest $q top 1 yield", bindings);
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(std::get<std::string>(results[0].data.at("tag")), "a");
}

TEST(EqlTest, FilteredScanViaEql) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    (void)db->query(R"(place in users vector [1.0, 0.0] data {"tier": "pro", "n": 1})");
    (void)db->query(R"(place in users vector [0.0, 1.0] data {"tier": "free", "n": 2})");
    (void)db->query(R"(place in users vector [1.0, 1.0] data {"tier": "pro", "n": 3})");

    const auto pros = db->query(R"(scan in users where tier = "pro" yield)");
    EXPECT_EQ(pros.size(), 2U);
}

TEST(EqlTest, FetchAndEraseViaEql) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    const auto inserted = db->query(R"(place in v vector [1.0, 2.0] data {"x": 1})");
    ASSERT_EQ(inserted.size(), 1U);
    const std::string id = inserted[0].id.to_string();

    const auto fetched = db->query("fetch from v id \"" + id + "\" yield");
    ASSERT_EQ(fetched.size(), 1U);

    (void)db->query("erase from v id \"" + id + "\"");
    EXPECT_EQ(db->vault("v").info().count, 0U);
}

TEST(EqlTest, InClauseAndBoolean) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    (void)db->query(R"(place in v vector [1.0, 0.0] data {"country": "US", "ok": true})");
    (void)db->query(R"(place in v vector [0.0, 1.0] data {"country": "FR", "ok": true})");
    const auto rows = db->query(
        R"(scan in v where country in ["US", "GB"] and ok = true yield)");
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(std::get<std::string>(rows[0].data.at("country")), "US");
}

TEST(EqlTest, MalformedQueryThrowsParseError) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    EXPECT_THROW((void)db->query("seek docs nearest"), elips::eql::ParseError);
}

}  // namespace
