#ifndef ELIPS_TEXT_ENGINE_LOCAL_TEXT_EMBEDDER_HPP
#define ELIPS_TEXT_ENGINE_LOCAL_TEXT_EMBEDDER_HPP

#include <memory>
#include <mutex>
#include <string>

#include "elips/text_engine/TextEmbedderPort.hpp"

namespace elips {

struct LocalTextEmbedderOptions {
    std::string model{"default"};
    std::string revision{"v1"};
    std::string storage_path;
    std::uint16_t dimension{0};
};

class LocalTextEmbedder final : public TextEmbedderPort {
public:
    explicit LocalTextEmbedder(LocalTextEmbedderOptions options);
    ~LocalTextEmbedder() override;

    [[nodiscard]] Vector embed(std::string_view text) const override;
    [[nodiscard]] std::string_view provider_name() const noexcept override;
    [[nodiscard]] std::string_view model_name() const noexcept override;
    [[nodiscard]] std::string_view revision_name() const noexcept override;
    [[nodiscard]] std::string_view backend_name() const noexcept override;
    [[nodiscard]] TextEmbedderKind kind() const noexcept override;
    [[nodiscard]] std::uint16_t output_dimension() const noexcept override;
    [[nodiscard]] bool rehydratable() const noexcept override;
    [[nodiscard]] bool loaded() const noexcept override;
    void unload() noexcept override;
    [[nodiscard]] std::string fingerprint() const override;
    [[nodiscard]] std::string storage_path() const override;

    [[nodiscard]] const LocalTextEmbedderOptions& options() const noexcept {
        return options_;
    }

private:
    struct Artifact;

    void ensure_loaded() const;

    LocalTextEmbedderOptions options_;
    std::string provider_{"elips-local"};
    std::string backend_{"builtin-hash"};
    std::string fingerprint_;
    mutable std::mutex mutex_;
    mutable std::unique_ptr<Artifact> artifact_;
};

[[nodiscard]] TextEmbedderInfo describe_local_text_embedder(
    const LocalTextEmbedderOptions& options,
    std::uint16_t fallback_dimension = 0,
    bool auto_attached = false);

[[nodiscard]] TextEmbedderPtr make_local_text_embedder(
    const LocalTextEmbedderOptions& options);

[[nodiscard]] TextEmbedderPtr make_default_local_text_embedder(
    std::uint16_t dimension, const std::string& storage_path = {});

}  // namespace elips

#endif  // ELIPS_TEXT_ENGINE_LOCAL_TEXT_EMBEDDER_HPP
