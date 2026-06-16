#ifndef ELIPS_CONFIG_HPP
#define ELIPS_CONFIG_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include "elips/text_engine/LocalTextEmbedder.hpp"
#include "elips/text_engine/TextEmbedderPort.hpp"

#ifdef ELIPS_GPU_ENABLED
#include "elips/gpu_engine/GpuConfig.hpp"
#endif

namespace elips {

class Config;
class ElipsInstance;
[[nodiscard]] std::unique_ptr<ElipsInstance> open(
    const std::string& path, const Config& config);

// Similarity metrics supported in v1.0. Scoped enum (Enum.3).
enum class Metric { cosine, euclidean, dot_product };

// Index backends. `exact` is brute-force (ground truth); `graph` is the
// HierarchicalGraphIndex (HNSW) primary ANN index.
enum class IndexType { graph, exact };

// Durability levels trade write throughput against crash safety.
//   paranoid  - flush WAL on every record
//   standard  - flush WAL on every record (group commit window is a future opt)
//   relaxed   - WAL buffered; flushed on checkpoint/close only
//   ephemeral - no WAL (the in-memory adapter uses this)
enum class Durability { paranoid, standard, relaxed, ephemeral };

enum class AccessMode { read_write, read_only };

[[nodiscard]] std::string_view to_string(Metric metric) noexcept;
[[nodiscard]] Metric metric_from_string(std::string_view name);

// Tunable parameters for the HierarchicalGraphIndex (HNSW).
struct GraphParams {
    std::size_t max_connections{16};    // M
    std::size_t ef_construction{200};   // beam width during build
    std::size_t ef_search{50};          // beam width during search
};

// Typed, validated configuration. Fluent builder; immutable intent after open().
class Config {
public:
    Config& dimension(std::uint16_t dim) noexcept {
        dimension_ = dim;
        return *this;
    }
    Config& metric(Metric metric) noexcept {
        metric_ = metric;
        return *this;
    }
    Config& index(IndexType type) noexcept {
        index_ = type;
        return *this;
    }
    Config& graph_params(GraphParams params) noexcept {
        graph_ = params;
        return *this;
    }
    Config& durability(Durability level) noexcept {
        durability_ = level;
        return *this;
    }
    Config& access_mode(AccessMode mode) noexcept {
        access_mode_ = mode;
        return *this;
    }
    Config& segmented_storage(bool enabled) noexcept {
        segmented_storage_ = enabled;
        return *this;
    }
    Config& metadata_acceleration(bool enabled) noexcept {
        metadata_acceleration_ = enabled;
        return *this;
    }
    Config& text_embedder(TextEmbedderPtr embedder) noexcept {
        text_embedder_ = std::move(embedder);
        local_text_embedder_.reset();
        expected_text_embedder_.reset();
        return *this;
    }
    Config& local_text_embedder(LocalTextEmbedderOptions options = {}) {
        local_text_embedder_ = std::move(options);
        text_embedder_ = {};
        expected_text_embedder_.reset();
        return *this;
    }
    Config& auto_text_embedder(bool enabled) noexcept {
        auto_text_embedder_ = enabled;
        return *this;
    }
#ifdef ELIPS_GPU_ENABLED
    Config& gpu(gpu::GpuConfig config) noexcept {
        gpu_ = std::move(config);
        return *this;
    }
#endif

    [[nodiscard]] std::uint16_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] Metric metric() const noexcept { return metric_; }
    [[nodiscard]] IndexType index() const noexcept { return index_; }
    [[nodiscard]] const GraphParams& graph_params() const noexcept { return graph_; }
    [[nodiscard]] Durability durability() const noexcept { return durability_; }
    [[nodiscard]] AccessMode access_mode() const noexcept { return access_mode_; }
    [[nodiscard]] bool segmented_storage() const noexcept { return segmented_storage_; }
    [[nodiscard]] bool metadata_acceleration() const noexcept { return metadata_acceleration_; }
    [[nodiscard]] const TextEmbedderPtr& text_embedder() const noexcept {
        return text_embedder_;
    }
    [[nodiscard]] bool has_text_embedder() const noexcept {
        return static_cast<bool>(text_embedder_) || local_text_embedder_.has_value();
    }
    [[nodiscard]] bool auto_text_embedder() const noexcept {
        return auto_text_embedder_;
    }
    [[nodiscard]] bool has_pending_local_text_embedder() const noexcept {
        return local_text_embedder_.has_value();
    }
    [[nodiscard]] const std::optional<LocalTextEmbedderOptions>&
    local_text_embedder_options() const noexcept {
        return local_text_embedder_;
    }
    [[nodiscard]] std::optional<TextEmbedderInfo> text_embedder_info() const {
        if (text_embedder_ != nullptr) {
            const bool auto_attached =
                expected_text_embedder_.has_value() &&
                expected_text_embedder_->auto_attached;
            return text_embedder_->info(auto_attached);
        }
        if (local_text_embedder_.has_value()) {
            return describe_local_text_embedder(
                *local_text_embedder_, dimension_, false);
        }
        return expected_text_embedder_;
    }
#ifdef ELIPS_GPU_ENABLED
    [[nodiscard]] const gpu::GpuConfig& gpu() const noexcept { return gpu_; }
    [[nodiscard]] bool has_gpu() const noexcept { return gpu_.policy != gpu::GpuPolicy::CpuOnly; }
#endif

    // Runtime resolver hooks used by open() to attach or remember the effective
    // text embedder after persisted identity has been reconciled.
    void attach_runtime_text_embedder(
        TextEmbedderPtr embedder,
        std::optional<TextEmbedderInfo> info = std::nullopt) noexcept {
        text_embedder_ = std::move(embedder);
        local_text_embedder_.reset();
        expected_text_embedder_ = std::move(info);
    }
    void remember_expected_text_embedder(TextEmbedderInfo info) noexcept {
        expected_text_embedder_ = std::move(info);
    }

private:
    friend std::unique_ptr<ElipsInstance> open(
        const std::string& path, const Config& config);

    std::uint16_t dimension_{0};
    Metric metric_{Metric::cosine};
    IndexType index_{IndexType::graph};
    GraphParams graph_{};
    Durability durability_{Durability::standard};
    AccessMode access_mode_{AccessMode::read_write};
    bool segmented_storage_{true};
    bool metadata_acceleration_{true};
    bool auto_text_embedder_{true};
    TextEmbedderPtr text_embedder_{};
    std::optional<LocalTextEmbedderOptions> local_text_embedder_{};
    std::optional<TextEmbedderInfo> expected_text_embedder_{};
#ifdef ELIPS_GPU_ENABLED
    gpu::GpuConfig gpu_{};
#endif
};

}  // namespace elips

#endif  // ELIPS_CONFIG_HPP
