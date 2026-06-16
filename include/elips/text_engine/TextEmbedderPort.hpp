#ifndef ELIPS_TEXT_ENGINE_TEXT_EMBEDDER_PORT_HPP
#define ELIPS_TEXT_ENGINE_TEXT_EMBEDDER_PORT_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "elips/domain/Vector.hpp"

namespace elips {

enum class TextEmbedderKind { external, local_builtin };

struct TextEmbedderInfo {
    TextEmbedderKind kind{TextEmbedderKind::external};
    std::string provider;
    std::string model;
    std::string revision;
    std::string backend;
    std::uint16_t dimension{0};
    std::string fingerprint;
    std::string storage_path;
    bool rehydratable{false};
    bool loaded{false};
    bool auto_attached{false};
};

class TextEmbedderPort {
public:
    virtual ~TextEmbedderPort() = default;

    [[nodiscard]] virtual Vector embed(std::string_view text) const = 0;

    [[nodiscard]] virtual std::vector<Vector> embed_batch(
        const std::vector<std::string>& texts) const {
        std::vector<Vector> embedded;
        embedded.reserve(texts.size());
        for (const auto& text : texts) {
            embedded.push_back(embed(text));
        }
        return embedded;
    }

    [[nodiscard]] virtual std::string_view provider_name() const noexcept = 0;
    [[nodiscard]] virtual std::string_view model_name() const noexcept = 0;
    [[nodiscard]] virtual std::string_view revision_name() const noexcept {
        return {};
    }
    [[nodiscard]] virtual std::string_view backend_name() const noexcept {
        return "custom";
    }
    [[nodiscard]] virtual TextEmbedderKind kind() const noexcept {
        return TextEmbedderKind::external;
    }
    [[nodiscard]] virtual std::uint16_t output_dimension() const noexcept {
        return 0;
    }
    [[nodiscard]] virtual bool rehydratable() const noexcept { return false; }
    [[nodiscard]] virtual bool loaded() const noexcept { return true; }
    virtual void unload() noexcept {}
    virtual void set_output_dimension(std::uint16_t) noexcept {}
    [[nodiscard]] virtual std::string fingerprint() const { return {}; }
    [[nodiscard]] virtual std::string storage_path() const { return {}; }

    [[nodiscard]] TextEmbedderInfo info(bool auto_attached = false) const {
        return TextEmbedderInfo{
            .kind = kind(),
            .provider = std::string(provider_name()),
            .model = std::string(model_name()),
            .revision = std::string(revision_name()),
            .backend = std::string(backend_name()),
            .dimension = output_dimension(),
            .fingerprint = fingerprint(),
            .storage_path = storage_path(),
            .rehydratable = rehydratable(),
            .loaded = loaded(),
            .auto_attached = auto_attached,
        };
    }
};

using TextEmbedderPtr = std::shared_ptr<TextEmbedderPort>;

}  // namespace elips

#endif  // ELIPS_TEXT_ENGINE_TEXT_EMBEDDER_PORT_HPP
