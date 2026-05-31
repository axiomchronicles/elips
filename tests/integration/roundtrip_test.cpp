#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "elips/domain/Errors.hpp"
#include "elips/elips.hpp"

namespace {

namespace fs = std::filesystem;

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
    auto& vault = db->vault("docs");
    const auto a = vault.place(elips::Vector{{1.0F, 0.0F, 0.0F}}, {{"k", std::string{"a"}}});
    vault.place(elips::Vector{{0.0F, 1.0F, 0.0F}}, {{"k", std::string{"b"}}});

    const auto results = vault.seek(elips::Vector{{0.9F, 0.1F, 0.0F}}, 1);
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].id, a);
    EXPECT_EQ(std::get<std::string>(results[0].data.at("k")), "a");
}

TEST_F(DatabaseTest, PersistenceRoundtripAcrossReopen) {
    elips::RecordID kept_id;
    {
        auto db = elips::open(path(), elips::Config{}.dimension(4).metric(
                                          elips::Metric::euclidean));
        auto& vault = db->vault("corpus");
        kept_id = vault.place(elips::Vector{{1.0F, 2.0F, 3.0F, 4.0F}},
                              {{"title", std::string{"hello"}},
                               {"year", std::int64_t{2024}}});
        vault.place(elips::Vector{{5.0F, 6.0F, 7.0F, 8.0F}});
        db->checkpoint();
    }  // destructor would checkpoint too, but we already did

    auto db = elips::open(path());  // reopen with no config: persisted identity wins
    EXPECT_EQ(db->config().dimension(), 4);
    EXPECT_EQ(db->config().metric(), elips::Metric::euclidean);

    auto& vault = db->vault("corpus");
    EXPECT_EQ(vault.info().count, 2U);

    const auto record = vault.fetch(kept_id);
    ASSERT_TRUE(record.has_value());
    EXPECT_EQ(std::get<std::string>(record->payload.at("title")), "hello");
    EXPECT_EQ(std::get<std::int64_t>(record->payload.at("year")), 2024);

    const auto hits = vault.seek(elips::Vector{{1.0F, 2.0F, 3.0F, 4.0F}}, 1);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].id, kept_id);
}

TEST_F(DatabaseTest, DimensionMismatchThrows) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(3));
    auto& vault = db->vault("v");
    EXPECT_THROW(vault.place(elips::Vector{{1.0F, 2.0F}}), elips::DimensionMismatch);
}

TEST_F(DatabaseTest, NonFiniteVectorRejected) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    auto& vault = db->vault("v");
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_THROW(vault.place(elips::Vector{{inf, 0.0F}}), elips::InvalidVector);
}

TEST_F(DatabaseTest, ReopenWithConflictingDimensionThrows) {
    { auto db = elips::open(path(), elips::Config{}.dimension(8)); db->checkpoint(); }
    EXPECT_THROW((void)elips::open(path(), elips::Config{}.dimension(16)),
                 elips::ConfigError);
}

TEST_F(DatabaseTest, EraseRemovesRecord) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2));
    auto& vault = db->vault("v");
    const auto id = vault.place(elips::Vector{{1.0F, 1.0F}});
    EXPECT_TRUE(vault.erase(id));
    EXPECT_FALSE(vault.fetch(id).has_value());
    EXPECT_EQ(vault.seek(elips::Vector{{1.0F, 1.0F}}, 5).size(), 0U);
}

TEST_F(DatabaseTest, AutoCheckpointOnDestruction) {
    {
        auto db = elips::open(path(), elips::Config{}.dimension(2));
        db->vault("v").place(elips::Vector{{1.0F, 2.0F}});
    }  // no explicit checkpoint; destructor must persist
    auto db = elips::open(path());
    EXPECT_EQ(db->vault("v").info().count, 1U);
}

}  // namespace
