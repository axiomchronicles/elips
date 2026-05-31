#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "elips/elips.hpp"
#include "elips/metadata/Filter.hpp"

namespace {

elips::Payload make_payload(const std::string& cat, std::int64_t year) {
    return {{"category", cat}, {"year", year}};
}

TEST(FilterTest, FilteredSeekRestrictsByMetadata) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2).metric(
                                          elips::Metric::euclidean));
    auto& v = db->vault("docs");
    v.place(elips::Vector{{0.0F, 0.0F}}, make_payload("tech", 2024));
    v.place(elips::Vector{{0.1F, 0.1F}}, make_payload("finance", 2024));
    v.place(elips::Vector{{0.2F, 0.2F}}, make_payload("tech", 2019));

    const auto filter =
        elips::Filter().field("category").equals(std::string{"tech"}).field("year").ge(
            std::int64_t{2020});
    const auto results = v.seek(elips::Vector{{0.0F, 0.0F}}, 10, filter);

    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(std::get<std::string>(results[0].data.at("category")), "tech");
    EXPECT_EQ(std::get<std::int64_t>(results[0].data.at("year")), 2024);
}

TEST(FilterTest, ScanWithFilterAndLimit) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    auto& v = db->vault("docs");
    for (int i = 0; i < 5; ++i) {
        v.place(elips::Vector{{static_cast<float>(i), 0.0F}},
                {{"keep", i % 2 == 0}});
    }
    const auto kept = v.scan(elips::Filter().field("keep").equals(true));
    EXPECT_EQ(kept.size(), 3U);  // i = 0, 2, 4
    EXPECT_EQ(v.scan(elips::Filter{}, /*offset=*/0, /*limit=*/2).size(), 2U);
}

TEST(FilterTest, OrAndNotComposition) {
    elips::Payload p{{"tier", std::string{"pro"}}, {"year", std::int64_t{2021}}};
    const auto pro = elips::Filter().field("tier").equals(std::string{"pro"});
    const auto recent = elips::Filter().field("year").ge(std::int64_t{2023});
    EXPECT_TRUE(pro.or_(recent).matches(p));
    EXPECT_FALSE(elips::Filter::not_(pro).matches(p));
}

TEST(TransactionTest, CommitAppliesAllWrites) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    {
        auto txn = db->begin_transaction();
        auto v = txn.vault("docs");
        v.place(elips::Vector{{1.0F, 0.0F}});
        v.place(elips::Vector{{0.0F, 1.0F}});
        txn.commit();
    }
    EXPECT_EQ(db->vault("docs").info().count, 2U);
}

TEST(TransactionTest, UncommittedTransactionRollsBack) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    {
        auto txn = db->begin_transaction();
        txn.vault("docs").place(elips::Vector{{1.0F, 0.0F}});
        // no commit: destructor rolls back
    }
    EXPECT_EQ(db->vault("docs").info().count, 0U);
}

TEST(TransactionTest, InvalidVectorRejectedAtBufferTime) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    auto txn = db->begin_transaction();
    EXPECT_THROW(txn.vault("docs").place(elips::Vector{{1.0F}}),
                 elips::DimensionMismatch);
}

}  // namespace
