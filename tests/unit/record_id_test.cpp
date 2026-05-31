#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <thread>

#include "elips/domain/RecordID.hpp"

namespace {

using elips::RecordID;

TEST(RecordIdTest, GenerateProducesVersion7Variant10) {
    const RecordID id = RecordID::generate();
    const auto& b = id.bytes();
    EXPECT_EQ(b[6] & 0xF0, 0x70);  // version 7
    EXPECT_EQ(b[8] & 0xC0, 0x80);  // variant 10
}

TEST(RecordIdTest, StringRoundTrip) {
    const RecordID id = RecordID::generate();
    const RecordID parsed = RecordID::from_string(id.to_string());
    EXPECT_EQ(id, parsed);
}

TEST(RecordIdTest, ToStringIsCanonicalHyphenatedForm) {
    const std::string s = RecordID::generate().to_string();
    ASSERT_EQ(s.size(), 36U);
    EXPECT_EQ(s[8], '-');
    EXPECT_EQ(s[13], '-');
    EXPECT_EQ(s[18], '-');
    EXPECT_EQ(s[23], '-');
}

TEST(RecordIdTest, TimePrefixIsMonotonicAcrossTime) {
    const RecordID earlier = RecordID::generate();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const RecordID later = RecordID::generate();
    // Compare the 48-bit timestamp prefix (bytes 0..5).
    EXPECT_LE(std::memcmp(earlier.bytes().data(), later.bytes().data(), 6), 0);
}

}  // namespace
