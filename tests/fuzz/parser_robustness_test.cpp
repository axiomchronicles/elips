// Parser robustness: feed mutated and adversarial bytes through WAL::replay()
// and the snapshot reader, asserting they fail fast and bounded rather than
// crashing, hanging, or over-allocating. Complements the libFuzzer target in
// tests/fuzz/, which explores the same surface without a fixed seed.
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "elips/elips.hpp"
#include "elips/storage/WAL.hpp"

namespace {

namespace fs = std::filesystem;

class ParserFuzzTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        dir_ = fs::temp_directory_path() / ("elips_fuzz_" + std::to_string(rd()));
        fs::create_directories(dir_);
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    fs::path dir_;

    // A WAL holding a few well-formed records of every shape.
    std::string valid_wal_bytes() {
        const fs::path path = dir_ / "seed.wal";
        {
            elips::WAL wal(path, /*sync_each_write=*/false);
            wal.append_insert("vault", elips::RecordID::generate(),
                              std::vector<float>{1.0F, 2.0F},
                              {{"k", std::string{"v"}},
                               {"n", std::int64_t{42}},
                               {"f", 1.5},
                               {"b", true}});
            wal.append_erase("vault", elips::RecordID::generate());
            elips::DocumentAttachment doc{
                .text = "body", .uri = "u", .mime_type = "text/plain"};
            elips::ChunkInfo chunk{.document_key = "d",
                                   .ordinal = 1,
                                   .char_start = 0,
                                   .char_end = 4};
            elips::EmbeddingLineage lineage{.provider = "p",
                                            .model = "m",
                                            .revision = "r",
                                            .attributes = {}};
            wal.append_txn_begin();
            wal.append_insert("vault", elips::RecordID::generate(),
                              std::vector<float>{3.0F, 4.0F}, {}, doc, chunk,
                              lineage);
            wal.append_txn_commit();
        }
        std::ifstream in(path, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    }

    void write_bytes(const fs::path& path, const std::string& bytes) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
};

TEST_F(ParserFuzzTest, RandomMutationsNeverCrashOrHang) {
    const std::string seed = valid_wal_bytes();
    ASSERT_FALSE(seed.empty());

    std::mt19937 rng(20260730);
    const fs::path path = dir_ / "mutated.wal";

    for (int iteration = 0; iteration < 2000; ++iteration) {
        std::string bytes = seed;
        // 1-4 single-byte mutations, biased toward the length-prefix positions
        // that F2 was about.
        const int mutations = 1 + static_cast<int>(rng() % 4);
        for (int m = 0; m < mutations; ++m) {
            const std::size_t pos = rng() % bytes.size();
            bytes[pos] = static_cast<char>(rng() % 256);
        }
        write_bytes(path, bytes);

        const auto start = std::chrono::steady_clock::now();
        // Must not throw out of replay(): recovery treats malformed input as a
        // corrupt tail and returns the valid prefix.
        const auto entries = elips::WAL::replay(path);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        EXPECT_LT(
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
            2000)
            << "replay took too long on mutated input at iteration " << iteration;
        // Whatever survives must be internally consistent.
        for (const auto& entry : entries) {
            EXPECT_LE(entry.vault.size(), bytes.size());
            EXPECT_LE(entry.vector.size(), bytes.size());
        }
    }
}

TEST_F(ParserFuzzTest, TruncationAtEveryOffsetIsHandled) {
    const std::string seed = valid_wal_bytes();
    const fs::path path = dir_ / "truncated.wal";

    for (std::size_t len = 0; len <= seed.size(); ++len) {
        write_bytes(path, seed.substr(0, len));
        const auto entries = elips::WAL::replay(path);
        // A prefix can only ever yield a prefix of the records.
        EXPECT_LE(entries.size(), 3U) << "at truncation length " << len;
    }
}

TEST_F(ParserFuzzTest, GarbageBytesProduceNoRecords) {
    std::mt19937 rng(7);
    const fs::path path = dir_ / "garbage.wal";
    for (int iteration = 0; iteration < 200; ++iteration) {
        std::string bytes(1 + (rng() % 4096), '\0');
        for (auto& byte : bytes) {
            byte = static_cast<char>(rng() % 256);
        }
        write_bytes(path, bytes);
        const auto entries = elips::WAL::replay(path);
        // Random bytes essentially never carry a valid magic + CRC pair.
        EXPECT_TRUE(entries.empty()) << "at iteration " << iteration;
    }
}

TEST_F(ParserFuzzTest, CorruptSnapshotIsRejectedNotCrashed) {
    // Build a real database, then corrupt its persisted state and reopen.
    const fs::path root = dir_ / "db";
    {
        auto db = elips::open(root.string(), elips::Config{}.dimension(4));
        for (int i = 0; i < 20; ++i) {
            db->vault("v").place(elips::Vector{
                {static_cast<float>(i), 1.0F, 2.0F, 3.0F}},
                {{"tag", std::string(16, 'x')}});
        }
        db->close();
    }

    fs::path segment;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() &&
            entry.path().extension() == ".segment") {
            segment = entry.path();
            break;
        }
    }
    ASSERT_FALSE(segment.empty()) << "expected a segment file to corrupt";

    std::ifstream in(segment, std::ios::binary);
    const std::string original((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
    in.close();

    std::mt19937 rng(99);
    for (int iteration = 0; iteration < 200; ++iteration) {
        std::string bytes = original;
        const std::size_t pos = rng() % bytes.size();
        bytes[pos] = static_cast<char>(rng() % 256);
        write_bytes(segment, bytes);

        const auto start = std::chrono::steady_clock::now();
        try {
            auto db = elips::open(root.string());
            // Opening may legitimately succeed if the flipped byte only altered
            // a vector value; the records must still be self-consistent.
            EXPECT_LE(db->vault("v").info().count, 20U);
            db->abandon();
        } catch (const elips::ElipsError&) {
            // A clear, typed rejection is the other acceptable outcome.
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        EXPECT_LT(
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
            2000)
            << "open() took too long on corrupt snapshot at iteration "
            << iteration;
    }
}

}  // namespace
