#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "elips/domain/Errors.hpp"
#include "elips/elips.hpp"
#include "elips/text_engine/TextEmbedderPort.hpp"

namespace {

namespace fs = std::filesystem;

class ToyEmbedder final : public elips::TextEmbedderPort {
public:
    [[nodiscard]] elips::Vector embed(std::string_view text) const override {
        const bool has_alpha = text.find("alpha") != std::string_view::npos;
        const bool has_beta = text.find("beta") != std::string_view::npos;
        return elips::Vector{{has_alpha ? 1.0F : 0.0F,
                              has_beta ? 1.0F : 0.0F}};
    }

    [[nodiscard]] std::string_view provider_name() const noexcept override {
        return "test";
    }

    [[nodiscard]] std::string_view model_name() const noexcept override {
        return "toy";
    }
};

// Fixture providing a unique, auto-cleaned database directory per test.
class DatabaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        dir_ = fs::temp_directory_path() /
               ("elips_it_" + std::to_string(rd()) + "_" + std::to_string(rd()));
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    std::string path() const { return dir_.string(); }

private:
    fs::path dir_;
};

TEST_F(DatabaseTest, InMemoryPlaceAndSeek) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(3).metric(
                                           elips::Metric::cosine));
    auto& collection = db->collection("docs");
    const auto a = collection.add(elips::Vector{{1.0F, 0.0F, 0.0F}}, {{"k", std::string{"a"}}});
    collection.add(elips::Vector{{0.0F, 1.0F, 0.0F}}, {{"k", std::string{"b"}}});

    const auto results = collection.search(elips::Vector{{0.9F, 0.1F, 0.0F}}, 1);
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].id, a);
    EXPECT_EQ(std::get<std::string>(results[0].data.at("k")), "a");
}

TEST_F(DatabaseTest, PersistenceRoundtripAcrossReopen) {
    elips::RecordID kept_id;
    {
        auto db = elips::open(path(), elips::Config{}.dimension(4).metric(
                                          elips::Metric::euclidean));
        auto& collection = db->collection("corpus");
        kept_id = collection.add(elips::Vector{{1.0F, 2.0F, 3.0F, 4.0F}},
                              {{"title", std::string{"hello"}},
                               {"year", std::int64_t{2024}}});
        collection.add(elips::Vector{{5.0F, 6.0F, 7.0F, 8.0F}});
        db->checkpoint();
    }  // destructor would checkpoint too, but we already did

    auto db = elips::open(path());  // reopen with no config: persisted identity wins
    EXPECT_EQ(db->config().dimension(), 4);
    EXPECT_EQ(db->config().metric(), elips::Metric::euclidean);

    auto& collection = db->collection("corpus");
    EXPECT_EQ(collection.info().count, 2U);

    const auto record = collection.get(kept_id);
    ASSERT_TRUE(record.has_value());
    EXPECT_EQ(std::get<std::string>(record->payload.at("title")), "hello");
    EXPECT_EQ(std::get<std::int64_t>(record->payload.at("year")), 2024);

    const auto hits = collection.search(elips::Vector{{1.0F, 2.0F, 3.0F, 4.0F}}, 1);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].id, kept_id);
}

TEST_F(DatabaseTest, DimensionMismatchThrows) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(3));
    auto& collection = db->collection("v");
    EXPECT_THROW(collection.add(elips::Vector{{1.0F, 2.0F}}), elips::DimensionMismatch);
}

TEST_F(DatabaseTest, NonFiniteVectorRejected) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    auto& collection = db->collection("v");
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_THROW(collection.add(elips::Vector{{inf, 0.0F}}), elips::InvalidVector);
}

TEST_F(DatabaseTest, ReopenWithConflictingDimensionThrows) {
    { auto db = elips::open(path(), elips::Config{}.dimension(8)); db->checkpoint(); }
    EXPECT_THROW((void)elips::open(path(), elips::Config{}.dimension(16)),
                 elips::ConfigError);
}

TEST_F(DatabaseTest, ReadOnlyOpenRequiresExistingDatabase) {
    EXPECT_THROW(
        (void)elips::open(
            path(),
            elips::Config{}.access_mode(elips::AccessMode::read_only)),
        elips::ConfigError);
}

TEST_F(DatabaseTest, EraseRemovesRecord) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    auto& collection = db->collection("v");
    const auto id = collection.add(elips::Vector{{1.0F, 1.0F}});
    EXPECT_TRUE(collection.erase(id));
    EXPECT_FALSE(collection.get(id).has_value());
    EXPECT_EQ(collection.search(elips::Vector{{1.0F, 1.0F}}, 5).size(), 0U);
}

TEST_F(DatabaseTest, ReplacingAnExistingIdKeepsSingleLogicalRecord) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2).index(
                                           elips::IndexType::exact));
    auto& collection = db->collection("v");
    const auto id = elips::RecordID::generate();

    collection.add(elips::Vector{{1.0F, 0.0F}},
                {{"title", std::string{"old"}}}, id);
    collection.add(elips::Vector{{0.0F, 1.0F}},
                {{"title", std::string{"new"}}}, id);

    EXPECT_EQ(collection.info().count, 1U);
    const auto fetched = collection.get(id);
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(std::get<std::string>(fetched->payload.at("title")), "new");

    const auto hits = collection.search(elips::Vector{{0.0F, 1.0F}}, 5);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].id, id);
    EXPECT_EQ(std::get<std::string>(hits[0].data.at("title")), "new");
}

TEST_F(DatabaseTest, AutoCheckpointOnDestruction) {
    {
        auto db = elips::open(path(), elips::Config{}.dimension(2));
        db->collection("v").add(elips::Vector{{1.0F, 2.0F}});
    }  // no explicit checkpoint; destructor must persist
    auto db = elips::open(path());
    EXPECT_EQ(db->collection("v").info().count, 1U);
}

TEST_F(DatabaseTest, DocumentLineageRoundTripAndNativeTextQuery) {
    auto db = elips::open(
        ":memory:",
        elips::Config{}
            .dimension(2)
            .text_embedder(std::make_shared<ToyEmbedder>()));
    auto& collection = db->collection("docs");

    elips::ChunkInfo chunk;
    chunk.document_key = "doc-alpha";
    chunk.ordinal = 3;
    chunk.char_start = 5;
    chunk.char_end = 14;

    const auto id = collection.add_document(
        "alpha note",
        {{"kind", std::string{"alpha"}}},
        std::nullopt,
        chunk);

    const auto record = collection.get(id);
    ASSERT_TRUE(record.has_value());
    ASSERT_TRUE(record->document.has_value());
    EXPECT_EQ(record->document->text, "alpha note");
    ASSERT_TRUE(record->chunk.has_value());
    EXPECT_EQ(record->chunk->document_key, "doc-alpha");
    EXPECT_EQ(record->chunk->ordinal, 3U);
    ASSERT_TRUE(record->lineage.has_value());
    EXPECT_EQ(record->lineage->provider, "test");
    EXPECT_EQ(record->lineage->model, "toy");

    const auto hits = collection.search_text("alpha", 1);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].id, id);
    ASSERT_TRUE(hits[0].document.has_value());
    EXPECT_EQ(hits[0].document->text, "alpha note");
}

TEST_F(DatabaseTest, ExplainSeekUsesMetadataAccelerationAndHybridPlanner) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    auto& collection = db->collection("docs");
    collection.add(elips::Vector{{1.0F, 0.0F}}, {{"kind", std::string{"alpha"}}});
    collection.add(elips::Vector{{0.0F, 1.0F}}, {{"kind", std::string{"beta"}}});

    const auto filter = elips::Filter().field("kind").equals("alpha");

    const auto exact_plan =
        collection.explain_query(elips::Vector{{1.0F, 0.0F}}, 1, filter);
    EXPECT_TRUE(exact_plan.metadata_accelerated);
    EXPECT_EQ(exact_plan.strategy, elips::QueryStrategy::exact_candidates);
    EXPECT_GE(exact_plan.candidate_count, 1U);

    const auto hybrid_plan =
        collection.explain_query(elips::Vector{{1.0F, 0.0F}}, 1, filter,
                           std::nullopt, true);
    EXPECT_TRUE(hybrid_plan.metadata_accelerated);
    EXPECT_EQ(hybrid_plan.strategy, elips::QueryStrategy::hybrid_fusion);
}

TEST_F(DatabaseTest, SegmentedStorageAndReadOnlyModeWorkTogether) {
    const fs::path root = path();
    {
        auto db = elips::open(
            root.string(),
            elips::Config{}.dimension(2).segmented_storage(true));
        auto& collection = db->collection("docs");
        collection.add(elips::Vector{{1.0F, 0.0F}},
                    {{"kind", std::string{"alpha"}}});
        collection.add(elips::Vector{{0.0F, 1.0F}},
                    {{"kind", std::string{"beta"}}});
        db->checkpoint();
    }

    EXPECT_TRUE(fs::exists(root / "elips.manifest"));
    EXPECT_TRUE(fs::exists(root / "segments"));

    {
        auto db = elips::open(root.string());
        EXPECT_EQ(db->collection("docs").info().count, 2U);
        db->compact();
    }

    auto reader_a = elips::open(
        root.string(),
        elips::Config{}.access_mode(elips::AccessMode::read_only));
    auto reader_b = elips::open(
        root.string(),
        elips::Config{}.access_mode(elips::AccessMode::read_only));

    EXPECT_EQ(reader_a->collection("docs").info().count, 2U);
    EXPECT_EQ(reader_b->collection("docs").info().count, 2U);
    EXPECT_THROW(
        reader_a->collection("docs").add(elips::Vector{{1.0F, 0.0F}}),
        elips::StorageError);
}

}  // namespace
