// M1 regression tests: durability (F1), bounded parser reads (F2), linear WAL
// replay (F3), atomic transaction commit (F4).
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "elips/elips.hpp"
#include "elips/storage/FileSync.hpp"
#include "elips/storage/Serialization.hpp"
#include "elips/storage/WAL.hpp"

namespace {

namespace fs = std::filesystem;

class M1Test : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        dir_ = fs::temp_directory_path() / ("elips_m1_" + std::to_string(rd()));
        fs::create_directories(dir_);
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
        elips::detail::sync_probe().fail_next.store(false);
        elips::detail::sync_probe().fail_after.store(0);
    }
    std::string path() const { return dir_.string(); }
    fs::path dir_;
};

// ------------------------------- F1: fsync -------------------------------

TEST_F(M1Test, WalAppendExercisesTheSyncPath) {
    auto& probe = elips::detail::sync_probe();
    const auto before = probe.calls.load();
    {
        elips::WAL wal(dir_ / "sync.wal", /*sync_each_write=*/true);
        wal.append_erase("v", elips::RecordID::generate());
    }
    EXPECT_GT(probe.calls.load(), before)
        << "append() must fsync, not just flush a stream buffer";
}

TEST_F(M1Test, RelaxedDurabilityDoesNotSyncPerAppend) {
    auto& probe = elips::detail::sync_probe();
    const auto before = probe.calls.load();
    {
        elips::WAL wal(dir_ / "relaxed.wal", /*sync_each_write=*/false);
        wal.append_erase("v", elips::RecordID::generate());
    }
    EXPECT_EQ(probe.calls.load(), before);
}

TEST_F(M1Test, FailedSyncSurfacesAsErrorNotSilentAck) {
    elips::WAL wal(dir_ / "faulty.wal", /*sync_each_write=*/true);
    elips::detail::sync_probe().fail_next.store(true);
    EXPECT_THROW(wal.append_erase("v", elips::RecordID::generate()),
                 elips::StorageError);
}

TEST_F(M1Test, CheckpointSyncsSnapshotBeforeAcknowledging) {
    auto& probe = elips::detail::sync_probe();
    auto db = elips::open(path(), elips::Config{}.dimension(2));
    db->vault("v").place(elips::Vector{{1.0F, 2.0F}});
    const auto before = probe.calls.load();
    db->checkpoint();
    EXPECT_GT(probe.calls.load(), before);
    db->abandon();
}

TEST_F(M1Test, WriteSurvivesAbruptTerminationAfterAppend) {
    // No checkpoint, no clean close: recovery depends purely on the synced WAL.
    {
        auto db = elips::open(path(), elips::Config{}.dimension(2).durability(
                                          elips::Durability::paranoid));
        db->vault("v").place(elips::Vector{{7.0F, 8.0F}});
        db->abandon();
    }
    auto db = elips::open(path());
    EXPECT_EQ(db->vault("v").info().count, 1U);
}

// --------------------------- F2: bounded reads ---------------------------

TEST_F(M1Test, OversizedStringLengthPrefixIsRejected) {
    std::string blob;
    const std::uint32_t bogus_len = 0xFFFFFF00U;  // ~4 GiB
    blob.append(reinterpret_cast<const char*>(&bogus_len), sizeof(bogus_len));
    blob.append("short");
    std::istringstream in(blob, std::ios::binary);
    EXPECT_THROW(elips::detail::get_string(in), elips::StorageError);
}

TEST_F(M1Test, OversizedPayloadCountIsRejected) {
    std::string blob;
    const std::uint32_t bogus_count = 0xFFFFFF00U;
    blob.append(reinterpret_cast<const char*>(&bogus_count), sizeof(bogus_count));
    std::istringstream in(blob, std::ios::binary);
    EXPECT_THROW(elips::detail::get_payload(in), elips::StorageError);
}

TEST_F(M1Test, WalWithGarbageLengthPrefixFailsFastAndBounded) {
    const fs::path wal_path = dir_ / "hostile.wal";
    {
        elips::WAL wal(wal_path);
        wal.append_insert("v", elips::RecordID::generate(),
                          std::vector<float>{1.0F, 2.0F}, {});
    }
    // Corrupt the vault-name length prefix (offset 5: magic 4 + op 1).
    {
        std::fstream out(wal_path,
                         std::ios::binary | std::ios::in | std::ios::out);
        out.seekp(5);
        const std::uint32_t bogus = 0xFFFFFF00U;
        out.write(reinterpret_cast<const char*>(&bogus), sizeof(bogus));
    }
    const auto start = std::chrono::steady_clock::now();
    const auto entries = elips::WAL::replay(wal_path);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_TRUE(entries.empty());
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(),
              5);
}

// -------------------------- F3: linear replay ---------------------------

TEST_F(M1Test, ReplayScalesLinearlyInRecordCount) {
    const auto time_replay = [&](std::size_t records, const char* name) {
        const fs::path wal_path = dir_ / name;
        {
            elips::WAL wal(wal_path, /*sync_each_write=*/false);
            for (std::size_t i = 0; i < records; ++i) {
                wal.append_insert("v", elips::RecordID::generate(),
                                  std::vector<float>(64, 1.0F),
                                  {{"k", std::string(64, 'x')}});
            }
        }
        const auto start = std::chrono::steady_clock::now();
        const auto entries = elips::WAL::replay(wal_path);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        EXPECT_EQ(entries.size(), records);
        return std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
            .count();
    };

    const auto small = std::max<std::int64_t>(time_replay(2000, "small.wal"), 1);
    const auto large = time_replay(8000, "large.wal");
    // 4x the records. Linear predicts ~4x; the old quadratic copy predicted
    // ~16x. Allow generous slack for timer noise on loaded machines.
    EXPECT_LT(large, small * 10)
        << "replay looks super-linear: " << small << "us -> " << large << "us";
}

// ------------------------ F4: atomic transactions -----------------------

TEST_F(M1Test, CommitOnReadOnlyVaultAppliesNothing) {
    auto db = elips::open(path(), elips::Config{}.dimension(2));
    auto& vault = db->vault("ro");
    const auto pre_existing = vault.place(elips::Vector{{9.0F, 9.0F}});
    vault.set_read_only(true);

    auto txn = db->begin_transaction();
    auto tv = txn.vault("ro");
    tv.place(elips::Vector{{1.0F, 0.0F}});
    tv.place(elips::Vector{{0.0F, 1.0F}});
    EXPECT_THROW(txn.commit(), elips::StorageError);

    vault.set_read_only(false);
    EXPECT_EQ(vault.info().count, 1U);
    EXPECT_TRUE(vault.fetch(pre_existing).has_value());
    db->abandon();
}

TEST_F(M1Test, CommitFailureMidBatchUndoesEarlierOps) {
    auto db = elips::open(path(), elips::Config{}.dimension(2).durability(
                                      elips::Durability::paranoid));
    auto& vault = db->vault("v");
    vault.place(elips::Vector{{5.0F, 5.0F}});
    ASSERT_EQ(vault.info().count, 1U);

    auto txn = db->begin_transaction();
    auto tv = txn.vault("v");
    for (int i = 0; i < 6; ++i) {
        tv.place(elips::Vector{{static_cast<float>(i), 1.0F}});
    }
    // Fail the 4th sync of the commit: txn_begin marker + 2 records land first,
    // so the undo path has real work to do.
    elips::detail::sync_probe().fail_after.store(4);
    EXPECT_THROW(txn.commit(), elips::StorageError);
    elips::detail::sync_probe().fail_after.store(0);

    // The pre-transaction record is intact and no batch record leaked in.
    EXPECT_EQ(vault.info().count, 1U);
    db->abandon();
}

TEST_F(M1Test, UncommittedWalRecordsAreDiscardedOnReplay) {
    const fs::path wal_path = dir_ / "txn.wal";
    const elips::RecordID committed = elips::RecordID::generate();
    const elips::RecordID orphan = elips::RecordID::generate();
    {
        elips::WAL wal(wal_path);
        wal.append_txn_begin();
        wal.append_insert("v", committed, std::vector<float>{1.0F, 0.0F}, {});
        wal.append_txn_commit();
        wal.append_txn_begin();
        wal.append_insert("v", orphan, std::vector<float>{0.0F, 1.0F}, {});
        // No commit marker: crash mid-batch.
    }
    const auto entries = elips::WAL::replay(wal_path);
    ASSERT_EQ(entries.size(), 1U);
    EXPECT_EQ(entries[0].id, committed);
}

TEST_F(M1Test, SuccessfulCommitStillAppliesEverything) {
    auto db = elips::open(path(), elips::Config{}.dimension(2));
    {
        auto txn = db->begin_transaction();
        auto tv = txn.vault("v");
        tv.place(elips::Vector{{1.0F, 0.0F}});
        tv.place(elips::Vector{{0.0F, 1.0F}});
        txn.commit();
    }
    EXPECT_EQ(db->vault("v").info().count, 2U);
    db->abandon();
}

TEST_F(M1Test, TransactionEraseRollsBackOnFailure) {
    auto db = elips::open(path(), elips::Config{}.dimension(2).durability(
                                      elips::Durability::paranoid));
    auto& vault = db->vault("v");
    const auto keep = vault.place(elips::Vector{{1.0F, 1.0F}});
    const auto also = vault.place(elips::Vector{{2.0F, 2.0F}});

    auto txn = db->begin_transaction();
    auto tv = txn.vault("v");
    tv.erase(keep);
    tv.erase(also);
    elips::detail::sync_probe().fail_next.store(true);
    EXPECT_THROW(txn.commit(), elips::StorageError);

    EXPECT_EQ(vault.info().count, 2U);
    EXPECT_TRUE(vault.fetch(keep).has_value());
    EXPECT_TRUE(vault.fetch(also).has_value());
    db->abandon();
}

}  // namespace
