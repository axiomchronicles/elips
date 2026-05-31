#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>

#include "elips/elips.hpp"
#include "elips/storage/WAL.hpp"

namespace {

namespace fs = std::filesystem;

class RecoveryTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        dir_ = fs::temp_directory_path() / ("elips_rec_" + std::to_string(rd()));
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    std::string path() const { return dir_.string(); }
    fs::path dir_;
};

TEST_F(RecoveryTest, WalReplayRecoversUncheckpointedWrites) {
    {
        auto db = elips::open(path(), elips::Config{}.dimension(3).durability(
                                          elips::Durability::paranoid));
        auto& vault = db->vault("v");
        vault.place(elips::Vector{{1.0F, 0.0F, 0.0F}}, {{"n", std::int64_t{1}}});
        vault.place(elips::Vector{{0.0F, 1.0F, 0.0F}}, {{"n", std::int64_t{2}}});
        db->abandon();  // simulate crash: no checkpoint, WAL persists
    }
    auto db = elips::open(path());
    EXPECT_EQ(db->vault("v").info().count, 2U);
    EXPECT_EQ(db->vault("v").seek(elips::Vector{{1.0F, 0.0F, 0.0F}}, 5).size(), 2U);
}

TEST_F(RecoveryTest, CorruptWalTailIsTruncatedNotFatal) {
    {
        auto db = elips::open(path(), elips::Config{}.dimension(2));
        db->vault("v").place(elips::Vector{{1.0F, 2.0F}});
        db->abandon();
    }
    // Append garbage to the end of the WAL.
    {
        std::ofstream wal(dir_ / "wal.log", std::ios::binary | std::ios::app);
        const char junk[] = {'\x01', '\x02', '\x03', '\x04', '\x05'};
        wal.write(junk, sizeof(junk));
    }
    auto db = elips::open(path());  // must not throw; recovers the valid record
    EXPECT_EQ(db->vault("v").info().count, 1U);
}

TEST_F(RecoveryTest, WalAppendAndReplayRoundTrip) {
    const fs::path wal_path = dir_ / "direct.wal";
    fs::create_directories(dir_);
    const elips::RecordID id = elips::RecordID::generate();
    {
        elips::WAL wal(wal_path);
        wal.append_insert("vault_a", id, std::vector<float>{1.0F, 2.0F},
                          {{"k", std::string{"v"}}});
        wal.append_erase("vault_a", elips::RecordID::generate());
    }
    const auto entries = elips::WAL::replay(wal_path);
    ASSERT_EQ(entries.size(), 2U);
    EXPECT_EQ(entries[0].op, elips::WAL::Op::insert);
    EXPECT_EQ(entries[0].id, id);
    EXPECT_EQ(entries[0].vault, "vault_a");
    EXPECT_EQ(entries[1].op, elips::WAL::Op::erase);
}

}  // namespace
