#include <gtest/gtest.h>

#include <cstdint>
#include <algorithm>
#include <string>

#include "elips/elips.hpp"

namespace {

TEST(ScanTest, EmptyVaultScanReturnsEmpty) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    auto& v = db->vault("docs");
    auto rows = v.scan();
    EXPECT_TRUE(rows.empty());
}

TEST(ScanTest, ScanReturnsAllRecords) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    auto& v = db->vault("docs");
    for (int i = 0; i < 10; ++i)
        v.place(elips::Vector{{static_cast<float>(i), 0.0F}}, {{"idx", std::int64_t{i}}});
    auto rows = v.scan();
    EXPECT_EQ(rows.size(), 10U);
}

TEST(ScanTest, ScanWithOffsetReturnsFewer) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(1));
    auto& v = db->vault("docs");
    for (int i = 0; i < 5; ++i)
        v.place(elips::Vector{{static_cast<float>(i)}}, {{"n", std::int64_t{i}}});
    auto rows = v.scan(elips::Filter{}, /*offset=*/3);
    ASSERT_EQ(rows.size(), 2U);
}

TEST(ScanTest, ScanWithLimitTruncates) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(1));
    auto& v = db->vault("docs");
    for (int i = 0; i < 10; ++i)
        v.place(elips::Vector{{static_cast<float>(i)}});
    auto rows = v.scan(elips::Filter{}, /*offset=*/0, /*limit=*/3);
    EXPECT_EQ(rows.size(), 3U);
}

TEST(ScanTest, ScanWithFilteredSearch) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    auto& v = db->vault("docs");
    v.place(elips::Vector{{0.0F, 0.0F}}, {{"tag", std::string{"red"}}});
    v.place(elips::Vector{{1.0F, 1.0F}}, {{"tag", std::string{"blue"}}});
    v.place(elips::Vector{{2.0F, 2.0F}}, {{"tag", std::string{"red"}}});

    auto reds = v.scan(elips::Filter().field("tag").equals(std::string{"red"}));
    EXPECT_EQ(reds.size(), 2U);

    auto blues = v.scan(elips::Filter().field("tag").equals(std::string{"blue"}));
    EXPECT_EQ(blues.size(), 1U);

    auto greens = v.scan(elips::Filter().field("tag").equals(std::string{"green"}));
    EXPECT_EQ(greens.size(), 0U);
}

TEST(ScanTest, ScanOffsetBeyondSizeReturnsEmpty) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(1));
    auto& v = db->vault("docs");
    v.place(elips::Vector{{1.0F}});
    v.place(elips::Vector{{2.0F}});
    auto rows = v.scan(elips::Filter{}, /*offset=*/10, /*limit=*/100);
    EXPECT_TRUE(rows.empty());
}

TEST(ScanTest, LargeLimitReturnsAll) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(1));
    auto& v = db->vault("docs");
    for (int i = 0; i < 50; ++i)
        v.place(elips::Vector{{static_cast<float>(i)}});
    auto rows = v.scan(elips::Filter{}, 0, 999999);
    EXPECT_EQ(rows.size(), 50U);
}

TEST(ScanTest, ScanWithMixedFilterAndLimitOffset) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(1));
    auto& v = db->vault("docs");
    for (int i = 0; i < 20; ++i)
        v.place(elips::Vector{{static_cast<float>(i)}}, {{"ok", i % 2 == 0}});

    auto filtered = v.scan(elips::Filter().field("ok").equals(true));
    EXPECT_EQ(filtered.size(), 10U);

    auto filtered_offset = v.scan(elips::Filter().field("ok").equals(true), 5, 3);
    EXPECT_EQ(filtered_offset.size(), 3U);
}

}  // namespace