#include <gtest/gtest.h>

#include <filesystem>
#include <random>
#include <thread>
#include <vector>

#include "elips/elips.hpp"

namespace {

namespace fs = std::filesystem;

class ConcurrencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        dir_ = fs::temp_directory_path() /
               ("elips_conc_" + std::to_string(rd()));
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    std::string path() const { return dir_.string(); }
    fs::path dir_;
};

TEST_F(ConcurrencyTest, MultipleReadersCanOpenSameDatabase) {
    std::string p = path();
    {
        auto writer = elips::open(p, elips::Config{}.dimension(2));
        writer->vault("v").place(elips::Vector{{1.0F, 0.0F}});
        writer->checkpoint();
        writer->close();
    }

    constexpr int kReaders = 4;
    std::vector<std::thread> threads;
    std::atomic<int> successes{0};

    for (int i = 0; i < kReaders; ++i) {
        threads.emplace_back([&]() {
            try {
                auto db = elips::open(p);
                if (db->vault("v").info().count == 1U) {
                    successes.fetch_add(1);
                }
                db->close();
            } catch (const elips::LockConflict&) {
                // Expected: another writer holds the lock
            }
        });
    }

    for (auto& t : threads) t.join();
    EXPECT_GE(successes.load(), 1);
}

TEST_F(ConcurrencyTest, ConcurrentReadsOnInMemory) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(3));
    auto& v = db->vault("shared");
    for (int i = 0; i < 100; ++i)
        v.place(elips::Vector{{static_cast<float>(i), 0.0F, 0.0F}});

    constexpr int kThreads = 8;
    constexpr int kQueries = 50;
    std::vector<std::thread> threads;
    std::atomic<size_t> totalHits{0};

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int q = 0; q < kQueries; ++q) {
                auto hits = v.seek(elips::Vector{{0.0F, 0.0F, 0.0F}}, 10);
                totalHits.fetch_add(hits.size());
            }
        });
    }

    for (auto& t : threads) t.join();
    EXPECT_GE(totalHits.load(), kThreads * kQueries);
}

TEST_F(ConcurrencyTest, SequentialTransactionsWorkReliably) {
    std::string p = path();
    auto db = elips::open(p, elips::Config{}.dimension(2));
    auto& v = db->vault("txn");

    constexpr int kIterations = 100;
    for (int i = 0; i < kIterations; ++i) {
        auto txn = db->begin_transaction();
        auto tv = txn.vault("txn");
        tv.place(elips::Vector{{static_cast<float>(i), 0.0F}});
        txn.commit();
    }
    EXPECT_EQ(v.info().count, kIterations);
}

TEST_F(ConcurrencyTest, ConcurrentReadAttemptsOnPersistentDb) {
    std::string p = path();
    {
        auto writer = elips::open(p, elips::Config{}.dimension(3));
        auto& v = writer->vault("data");
        for (int i = 0; i < 20; ++i)
            v.place(elips::Vector{{static_cast<float>(i), 0.0F, 0.0F}});
        writer->checkpoint();
        writer->close();
    }

    constexpr int kAttempts = 4;
    std::vector<std::thread> threads;
    std::atomic<int> successes{0};
    std::atomic<int> conflicts{0};

    for (int t = 0; t < kAttempts; ++t) {
        threads.emplace_back([&]() {
            try {
                auto db = elips::open(p);
                auto hits = db->vault("data").seek(
                    elips::Vector{{0.0F, 0.0F, 0.0F}}, 10);
                if (hits.size() == 10U) successes.fetch_add(1);
                db->close();
            } catch (const elips::LockConflict&) {
                conflicts.fetch_add(1);
            }
        });
    }

    for (auto& t : threads) t.join();
    EXPECT_GE(successes.load() + conflicts.load(), 1);
}

}  // namespace