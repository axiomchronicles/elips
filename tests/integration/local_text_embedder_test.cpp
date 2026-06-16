#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "elips/domain/Errors.hpp"
#include "elips/elips.hpp"
#include "elips/text_engine/LocalTextEmbedder.hpp"
#include "elips/text_engine/TextEmbedderPort.hpp"

namespace {

namespace fs = std::filesystem;

class ExternalToyEmbedder final : public elips::TextEmbedderPort {
public:
    [[nodiscard]] elips::Vector embed(std::string_view text) const override {
        const std::string lowered{text};
        return elips::Vector{{
            lowered.find("alpha") != std::string::npos ? 1.0F : 0.0F,
            lowered.find("beta") != std::string::npos ? 1.0F : 0.0F,
            lowered.find("database") != std::string::npos ? 1.0F : 0.0F,
            lowered.find("python") != std::string::npos ? 1.0F : 0.0F,
        }};
    }

    [[nodiscard]] std::string_view provider_name() const noexcept override {
        return "test";
    }

    [[nodiscard]] std::string_view model_name() const noexcept override {
        return "toy";
    }

    [[nodiscard]] std::string_view revision_name() const noexcept override {
        return "r1";
    }

    [[nodiscard]] std::string_view backend_name() const noexcept override {
        return "test-backend";
    }

    [[nodiscard]] std::uint16_t output_dimension() const noexcept override {
        return 4U;
    }
};

class LocalTextEmbedderTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        dir_ = fs::temp_directory_path() /
               ("elips_local_embedder_" + std::to_string(rd()) + "_" +
                std::to_string(rd()));
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    [[nodiscard]] std::string path() const { return dir_.string(); }

    [[nodiscard]] fs::path root() const { return dir_; }

private:
    fs::path dir_;
};

std::vector<float> materialize(std::span<const float> values) {
    return std::vector<float>(values.begin(), values.end());
}

TEST_F(LocalTextEmbedderTest, LocalEmbedderLoadsLazilyAndRemainsDeterministic) {
    const fs::path artifact = root() / "model.localembed";

    elips::LocalTextEmbedderOptions options;
    options.model = "default";
    options.revision = "v1";
    options.storage_path = artifact.string();
    options.dimension = 64;

    const auto described = elips::describe_local_text_embedder(options);
    EXPECT_EQ(described.kind, elips::TextEmbedderKind::local_builtin);
    EXPECT_EQ(described.dimension, 64U);
    EXPECT_FALSE(described.loaded);
    EXPECT_EQ(described.provider, "elips-local");

    const auto runtime = elips::make_local_text_embedder(options);
    auto* local =
        dynamic_cast<elips::LocalTextEmbedder*>(runtime.get());
    ASSERT_NE(local, nullptr);
    EXPECT_FALSE(local->loaded());
    EXPECT_TRUE(fs::exists(artifact));

    const auto first = materialize(local->embed("hello vector database").values());
    EXPECT_TRUE(local->loaded());
    local->unload();
    EXPECT_FALSE(local->loaded());
    const auto second = materialize(local->embed("hello vector database").values());

    EXPECT_EQ(first, second);
}

TEST_F(LocalTextEmbedderTest, NewDatabaseAutoAttachesAndRehydratesDefaultLocalEmbedder) {
    std::string first_id;
    std::string artifact_path;
    {
        auto db = elips::open(
            path(),
            elips::Config{}.dimension(64).metric(elips::Metric::cosine));
        ASSERT_TRUE(db->config().has_text_embedder());
        const auto info = db->config().text_embedder_info();
        ASSERT_TRUE(info.has_value());
        EXPECT_TRUE(info->auto_attached);
        EXPECT_TRUE(info->rehydratable);
        EXPECT_EQ(info->kind, elips::TextEmbedderKind::local_builtin);
        artifact_path = info->storage_path;

        auto& docs = db->vault("docs");
        first_id = docs.place_document(
            "vector database handbook",
            {{"kind", std::string{"db"}}})
                       .to_string();
        docs.place_document("python language guide",
                            {{"kind", std::string{"py"}}});
        db->checkpoint();
    }

    EXPECT_TRUE(fs::exists(root() / "TEXT_EMBEDDER.manifest"));
    EXPECT_TRUE(fs::exists(artifact_path));

    auto reopened = elips::open(path());
    ASSERT_TRUE(reopened->config().has_text_embedder());
    const auto info = reopened->config().text_embedder_info();
    ASSERT_TRUE(info.has_value());
    EXPECT_TRUE(info->auto_attached);
    EXPECT_EQ(info->storage_path, artifact_path);

    const auto hits = reopened->vault("docs").seek_text("vector database", 1);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].id.to_string(), first_id);
}

TEST_F(LocalTextEmbedderTest, MissingLocalArtifactFailsOnReopen) {
    std::string artifact_path;
    {
        auto db = elips::open(
            path(),
            elips::Config{}.dimension(64).metric(elips::Metric::cosine));
        const auto info = db->config().text_embedder_info();
        ASSERT_TRUE(info.has_value());
        artifact_path = info->storage_path;
        db->checkpoint();
    }

    ASSERT_TRUE(fs::remove(artifact_path));
    EXPECT_THROW((void)elips::open(path()), elips::StorageError);
}

TEST_F(LocalTextEmbedderTest, TextFirstApisExplainHowToRecoverWhenDisabled) {
    auto db = elips::open(
        ":memory:",
        elips::Config{}
            .dimension(64)
            .metric(elips::Metric::cosine)
            .auto_text_embedder(false));
    auto& docs = db->vault("docs");

    try {
        (void)docs.place_document("alpha note");
        FAIL() << "place_document should require a text embedder";
    } catch (const elips::ConfigError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("place()"), std::string::npos);
        EXPECT_NE(message.find("local text embedder"), std::string::npos);
    }

    std::vector<float> raw(64U, 0.0F);
    raw.front() = 1.0F;
    docs.place(elips::Vector{std::move(raw)},
               {},
               std::nullopt,
               elips::DocumentAttachment{.text = "vector only note"});

    try {
        (void)docs.seek_text("alpha", 1);
        FAIL() << "seek_text should require a text embedder";
    } catch (const elips::ConfigError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("seek()"), std::string::npos);
    }
}

TEST_F(LocalTextEmbedderTest, ExternalEmbedderMetadataRequiresExplicitReopen) {
    {
        auto db = elips::open(
            path(),
            elips::Config{}
                .dimension(4)
                .metric(elips::Metric::cosine)
                .text_embedder(std::make_shared<ExternalToyEmbedder>()));
        db->vault("docs").place_document("alpha database");
        db->checkpoint();
    }

    {
        auto reopened = elips::open(path());
        EXPECT_FALSE(reopened->config().has_text_embedder());
        const auto info = reopened->config().text_embedder_info();
        ASSERT_TRUE(info.has_value());
        EXPECT_EQ(info->provider, "test");
        EXPECT_EQ(info->model, "toy");
        EXPECT_FALSE(info->rehydratable);

        try {
            (void)reopened->vault("docs").place_document("beta database");
            FAIL() << "reopen without explicit external embedder should fail";
        } catch (const elips::ConfigError& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("matching embedder"), std::string::npos);
        }
    }

    auto explicit_reopen = elips::open(
        path(),
        elips::Config{}
            .dimension(4)
            .metric(elips::Metric::cosine)
            .text_embedder(std::make_shared<ExternalToyEmbedder>()));
    EXPECT_TRUE(explicit_reopen->config().has_text_embedder());
    const auto hits = explicit_reopen->vault("docs").seek_text("alpha", 1);
    ASSERT_EQ(hits.size(), 1U);
}

}  // namespace
