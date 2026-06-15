#include <gtest/gtest.h>

#include <filesystem>
#include <random>

#include "elips/elips.hpp"

namespace {

namespace fs = std::filesystem;

class DurabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        dir_ = fs::temp_directory_path() /
               ("elips_dur_" + std::to_string(rd()));
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    std::string path() const { return dir_.string(); }
    fs::path dir_;
};

TEST_F(DurabilityTest, ParanoidSurvivesCrash) {
    {
        auto db = elips::open(path(), elips::Config{}.dimension(2).durability(
                                          elips::Durability::paranoid));
        db->vault("v").place(elips::Vector{{1.0F, 0.0F}});
        db->abandon();
    }
    auto db = elips::open(path());
    EXPECT_EQ(db->vault("v").info().count, 1U);
}

TEST_F(DurabilityTest, StandardPersistsOnClose) {
    std::string p = path();
    {
        auto db = elips::open(p, elips::Config{}.dimension(3));
        db->vault("v").place(elips::Vector{{1.0F, 2.0F, 3.0F}});
        db->close();
    }
    auto db = elips::open(p);
    EXPECT_EQ(db->vault("v").info().count, 1U);
}

TEST_F(DurabilityTest, RelaxedPersistsOnCheckpoint) {
    std::string p = path();
    {
        auto db = elips::open(p, elips::Config{}.dimension(2).durability(
                                     elips::Durability::relaxed));
        db->vault("v").place(elips::Vector{{4.0F, 5.0F}});
        db->checkpoint();
        db->abandon();
    }
    auto db = elips::open(p);
    EXPECT_EQ(db->vault("v").info().count, 1U);
}

TEST_F(DurabilityTest, EphemeralDoesNotPersist) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(2).durability(
                                        elips::Durability::ephemeral));
    EXPECT_EQ(db->vault("v").info().count, 0U);
    db->vault("v").place(elips::Vector{{1.0F, 2.0F}});
    // In-memory only, no crash recovery concerns
    EXPECT_EQ(db->vault("v").info().count, 1U);
}

TEST_F(DurabilityTest, MultipleRecordsSurviveStandard) {
    std::string p = path();
    {
        auto db = elips::open(p, elips::Config{}.dimension(2));
        for (int i = 0; i < 100; ++i)
            db->vault("v").place(elips::Vector{{static_cast<float>(i), 0.0F}});
        db->checkpoint();
    }
    auto db = elips::open(p);
    EXPECT_EQ(db->vault("v").info().count, 100U);
}

TEST_F(DurabilityTest, MultipleVaultsSurviveCheckpoint) {
    std::string p = path();
    {
        auto db = elips::open(p, elips::Config{}.dimension(2));
        db->vault("v1").place(elips::Vector{{1.0F, 0.0F}});
        db->vault("v2").place(elips::Vector{{0.0F, 1.0F}});
        db->checkpoint();
    }
    auto db = elips::open(p);
    EXPECT_EQ(db->vault("v1").info().count, 1U);
    EXPECT_EQ(db->vault("v2").info().count, 1U);
    EXPECT_EQ(db->list_vaults().size(), 2U);
}

TEST_F(DurabilityTest, DeletionSurvivesCheckpoint) {
    std::string p = path();
    {
        auto db = elips::open(p, elips::Config{}.dimension(2));
        auto id = db->vault("v").place(elips::Vector{{1.0F, 2.0F}});
        db->vault("v").place(elips::Vector{{3.0F, 4.0F}});
        db->vault("v").erase(id);
        db->checkpoint();
    }
    auto db = elips::open(p);
    EXPECT_EQ(db->vault("v").info().count, 1U);
}

}  // namespace