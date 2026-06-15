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

TEST_F(RecoveryTest, WalReplayRecoversExtendedRecordState) {
    elips::RecordID kept_id;
    {
        auto db = elips::open(path(), elips::Config{}.dimension(2).durability(
                                          elips::Durability::paranoid));
        auto& vault = db->vault("docs");

        elips::DocumentAttachment document;
        document.text = "alpha note";
        document.uri = "file:///tmp/alpha.txt";
        document.mime_type = "text/plain";

        elips::ChunkInfo chunk;
        chunk.document_key = "doc-alpha";
        chunk.ordinal = 1;
        chunk.char_start = 3;
        chunk.char_end = 12;

        elips::EmbeddingLineage lineage;
        lineage.provider = "pytest";
        lineage.model = "toy";
        lineage.revision = "v2";
        lineage.attributes = {{"stage", std::string{"recovery"}}};

        kept_id = vault.place(elips::Vector{{1.0F, 0.0F}},
                              {{"kind", std::string{"alpha"}}},
                              std::nullopt,
                              document,
                              chunk,
                              lineage);
        db->abandon();
    }

    auto recovered = elips::open(path());
    const auto record = recovered->vault("docs").fetch(kept_id);
    ASSERT_TRUE(record.has_value());
    ASSERT_TRUE(record->document.has_value());
    EXPECT_EQ(record->document->text, "alpha note");
    EXPECT_EQ(record->document->uri, "file:///tmp/alpha.txt");
    ASSERT_TRUE(record->chunk.has_value());
    EXPECT_EQ(record->chunk->document_key, "doc-alpha");
    EXPECT_EQ(record->chunk->ordinal, 1U);
    ASSERT_TRUE(record->lineage.has_value());
    EXPECT_EQ(record->lineage->provider, "pytest");
    EXPECT_EQ(record->lineage->model, "toy");
    EXPECT_EQ(std::get<std::string>(record->lineage->attributes.at("stage")),
              "recovery");
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

TEST_F(RecoveryTest, WalAppendAndReplayExtendedInsertRoundTrip) {
    const fs::path wal_path = dir_ / "extended.wal";
    fs::create_directories(dir_);
    const elips::RecordID id = elips::RecordID::generate();

    elips::DocumentAttachment document;
    document.text = "alpha note";
    document.uri = "memory://alpha";
    document.mime_type = "text/plain";

    elips::ChunkInfo chunk;
    chunk.document_key = "doc-alpha";
    chunk.ordinal = 2;
    chunk.char_start = 4;
    chunk.char_end = 13;

    elips::EmbeddingLineage lineage;
    lineage.provider = "pytest";
    lineage.model = "toy";
    lineage.revision = "v2";
    lineage.attributes = {{"stage", std::string{"wal"}}};

    {
        elips::WAL wal(wal_path);
        wal.append_insert("vault_docs", id, std::vector<float>{1.0F, 0.0F},
                          {{"kind", std::string{"alpha"}}}, document, chunk,
                          lineage);
    }

    const auto entries = elips::WAL::replay(wal_path);
    ASSERT_EQ(entries.size(), 1U);
    EXPECT_EQ(entries[0].op, elips::WAL::Op::insert_ex);
    EXPECT_EQ(entries[0].id, id);
    ASSERT_TRUE(entries[0].document.has_value());
    EXPECT_EQ(entries[0].document->text, "alpha note");
    ASSERT_TRUE(entries[0].chunk.has_value());
    EXPECT_EQ(entries[0].chunk->document_key, "doc-alpha");
    ASSERT_TRUE(entries[0].lineage.has_value());
    EXPECT_EQ(entries[0].lineage->provider, "pytest");
    EXPECT_EQ(entries[0].lineage->model, "toy");
    EXPECT_EQ(std::get<std::string>(entries[0].lineage->attributes.at("stage")),
              "wal");
}

}  // namespace
