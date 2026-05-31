#ifndef ELIPS_CONFIG_HPP
#define ELIPS_CONFIG_HPP

#include <cstdint>
#include <string_view>

namespace elips {

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

    [[nodiscard]] std::uint16_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] Metric metric() const noexcept { return metric_; }
    [[nodiscard]] IndexType index() const noexcept { return index_; }
    [[nodiscard]] const GraphParams& graph_params() const noexcept { return graph_; }
    [[nodiscard]] Durability durability() const noexcept { return durability_; }

private:
    std::uint16_t dimension_{0};
    Metric metric_{Metric::cosine};
    IndexType index_{IndexType::graph};
    GraphParams graph_{};
    Durability durability_{Durability::standard};
};

}  // namespace elips

#endif  // ELIPS_CONFIG_HPP
