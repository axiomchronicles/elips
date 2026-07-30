// M2 regression tests: in-process reader/writer safety (F6) and post-close
// write guarding (F7). Run this binary under ThreadSanitizer to validate F6:
//   cmake -S . -B build-tsan -DELIPS_SANITIZE=thread && ctest --test-dir build-tsan
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

#include "elips/elips.hpp"

namespace {

namespace fs = std::filesystem;

class M2Test : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        dir_ = fs::temp_directory_path() / ("elips_m2_" + std::to_string(rd()));
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    std::string path() const { return dir_.string(); }
    fs::path dir_;
};

// -------------------------- F6: in-process locking -----------------------

TEST_F(M2Test, ConcurrentWriterAndReadersOnOneVault) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(8));
    auto& vault = db->vault("v");

    constexpr int kSeed = 32;
    for (int i = 0; i < kSeed; ++i) {
        vault.place(elips::Vector{std::vector<float>(8, static_cast<float>(i))});
    }

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> reads{0};
    std::atomic<std::uint64_t> writes{0};

    std::vector<std::thread> threads;
    // Two writers: exercises writer/writer exclusion, not just writer/reader.
    for (int w = 0; w < 2; ++w) {
        threads.emplace_back([&, w] {
            for (int i = 0; i < 200; ++i) {
                vault.place(elips::Vector{std::vector<float>(
                    8, static_cast<float>((w * 1000) + i))});
                writes.fetch_add(1);
            }
        });
    }
    for (int r = 0; r < 4; ++r) {
        threads.emplace_back([&] {
            const elips::Vector query{std::vector<float>(8, 1.0F)};
            while (!stop.load()) {
                (void)vault.seek(query, 5);
                (void)vault.info();
                (void)vault.scan({}, 0, 4);
                reads.fetch_add(1);
            }
        });
    }

    for (int w = 0; w < 2; ++w) {
        threads[static_cast<std::size_t>(w)].join();
    }
    stop.store(true);
    for (std::size_t i = 2; i < threads.size(); ++i) {
        threads[i].join();
    }

    EXPECT_EQ(writes.load(), 400U);
    EXPECT_GT(reads.load(), 0U);
    EXPECT_EQ(vault.info().count, static_cast<std::size_t>(kSeed) + 400U);
}

TEST_F(M2Test, ConcurrentEraseAndSearchStayConsistent) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(4));
    auto& vault = db->vault("v");

    std::vector<elips::RecordID> ids;
    ids.reserve(300);
    for (int i = 0; i < 300; ++i) {
        ids.push_back(vault.place(
            elips::Vector{std::vector<float>(4, static_cast<float>(i))}));
    }

    std::atomic<bool> stop{false};
    std::thread eraser([&] {
        for (const auto& id : ids) {
            vault.erase(id);
        }
        stop.store(true);
    });
    std::thread reader([&] {
        const elips::Vector query{std::vector<float>(4, 2.0F)};
        while (!stop.load()) {
            const auto hits = vault.seek(query, 10);
            EXPECT_LE(hits.size(), 10U);
            for (const auto& hit : hits) {
                (void)vault.fetch(hit.id);
            }
        }
    });
    eraser.join();
    reader.join();

    EXPECT_EQ(vault.info().count, 0U);
}

TEST_F(M2Test, ConcurrentVaultCreationIsSafe) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&db, t] {
            for (int i = 0; i < 20; ++i) {
                auto& vault = db->vault("vault_" + std::to_string(i));
                vault.place(elips::Vector{{static_cast<float>(t), 1.0F}});
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(db->list_vaults().size(), 20U);
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(db->vault("vault_" + std::to_string(i)).info().count, 8U);
    }
}

TEST_F(M2Test, ConcurrentTransactionsDoNotInterleave) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&db, t] {
            for (int batch = 0; batch < 10; ++batch) {
                auto txn = db->begin_transaction();
                auto tv = txn.vault("v");
                for (int i = 0; i < 5; ++i) {
                    tv.place(elips::Vector{
                        {static_cast<float>(t), static_cast<float>(batch)}});
                }
                txn.commit();
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(db->vault("v").info().count, 4U * 10U * 5U);
}

// ------------------------ F7: writes after close() -----------------------

TEST_F(M2Test, PlaceAfterCloseThrowsInsteadOfVanishing) {
    auto db = elips::open(path(), elips::Config{}.dimension(2));
    auto& vault = db->vault("v");
    vault.place(elips::Vector{{1.0F, 2.0F}});
    db->close();

    EXPECT_THROW(vault.place(elips::Vector{{3.0F, 4.0F}}), elips::StorageError);
}

TEST_F(M2Test, EraseAfterCloseThrows) {
    auto db = elips::open(path(), elips::Config{}.dimension(2));
    auto& vault = db->vault("v");
    const auto id = vault.place(elips::Vector{{1.0F, 2.0F}});
    db->close();

    EXPECT_THROW(vault.erase(id), elips::StorageError);
}

TEST_F(M2Test, VaultCreatedAfterCloseRefusesWrites) {
    auto db = elips::open(path(), elips::Config{}.dimension(2));
    db->vault("v").place(elips::Vector{{1.0F, 2.0F}});
    db->close();

    EXPECT_THROW(db->vault("later").place(elips::Vector{{5.0F, 6.0F}}),
                 elips::StorageError);
}

TEST_F(M2Test, TransactionCommitAfterCloseThrows) {
    auto db = elips::open(path(), elips::Config{}.dimension(2));
    auto txn = db->begin_transaction();
    auto tv = txn.vault("v");
    tv.place(elips::Vector{{1.0F, 2.0F}});
    db->close();

    EXPECT_THROW(txn.commit(), elips::StorageError);
}

TEST_F(M2Test, WriteRejectedAfterCloseIsAbsentOnReopen) {
    std::string p = path();
    {
        auto db = elips::open(p, elips::Config{}.dimension(2));
        auto& vault = db->vault("v");
        vault.place(elips::Vector{{1.0F, 2.0F}});
        db->close();
        EXPECT_THROW(vault.place(elips::Vector{{9.0F, 9.0F}}),
                     elips::StorageError);
    }
    auto db = elips::open(p);
    EXPECT_EQ(db->vault("v").info().count, 1U);
}

TEST_F(M2Test, ReadsAfterCloseStillWork) {
    auto db = elips::open(path(), elips::Config{}.dimension(2));
    auto& vault = db->vault("v");
    const auto id = vault.place(elips::Vector{{1.0F, 0.0F}});
    db->close();

    // Reads are served from memory that is still valid and consistent; only
    // mutations are refused.
    EXPECT_EQ(vault.info().count, 1U);
    EXPECT_TRUE(vault.fetch(id).has_value());
    EXPECT_EQ(vault.seek(elips::Vector{{1.0F, 0.0F}}, 1).size(), 1U);
}

}  // namespace
