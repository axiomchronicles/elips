#ifndef ELIPS_INDEX_ENGINE_HIERARCHICAL_GRAPH_INDEX_HPP
#define ELIPS_INDEX_ENGINE_HIERARCHICAL_GRAPH_INDEX_HPP

#include <cstddef>
#include <cstdint>
#include "elips/domain/Expected.hpp"
#include <random>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/IndexPort.hpp"
#include "elips/index_engine/IndexTransferPort.hpp"
#include "elips/quant_engine/Quantizer.hpp"

namespace elips {

// Primary ANN index: a from-scratch Hierarchical Navigable Small World graph.
// Vectors are stored row-major; the graph is layered with probabilistic level
// assignment.
//
// Deletes are soft tombstones so graph navigation stays intact, but tombstones
// are not free: they occupy the search beam and their vectors and edges keep
// consuming memory. Two mechanisms bound that cost:
//   * search() widens `ef` in proportion to the live/deleted ratio, so a
//     tombstoned graph still returns k live results rather than silently fewer.
//   * once tombstones exceed `compaction_ratio` of the graph the index rebuilds
//     itself (vacuum()), reclaiming dead vectors and edges. Callers can force
//     this at any time via vacuum().
//
// With a trained quantizer attached, node vectors are stored as codes and every
// distance -- during traversal as well as at query time -- is computed by
// asymmetric lookup. Compression therefore has to reach inside the graph rather
// than wrap it: HNSW evaluates distances while navigating, so a wrapper around
// search() could not compress the storage those evaluations read.
class HierarchicalGraphIndex final : public IndexPort, public IndexTransferPort {
public:
    HierarchicalGraphIndex(Metric metric, std::uint16_t dimension,
                           GraphParams params,
                           quant::QuantizerPtr quantizer = {});

    void insert(const RecordID& id, std::span<const float> vector) override;
    void remove(const RecordID& id) override;
    [[nodiscard]] std::vector<Hit> search(std::span<const float> query,
                                          std::size_t k) const override;

    [[nodiscard]] std::size_t size() const noexcept override {
        return ids_.size() - deleted_count_;
    }
    [[nodiscard]] std::string_view type_name() const noexcept override {
        return quantizer_ != nullptr ? "graph_quantized" : "graph";
    }

    void vacuum() override;
    [[nodiscard]] std::size_t pending_removals() const noexcept override {
        return deleted_count_;
    }

    [[nodiscard]] std::expected<IndexSnapshot, std::string>
    export_snapshot() const override;
    [[nodiscard]] std::expected<void, std::string>
    import_snapshot(const IndexSnapshot& snapshot) override;

private:
    using NodeId = std::uint32_t;
    using Scored = std::pair<float, NodeId>;  // (distance, node)

    // Distance evaluator bound to one query for the lifetime of a traversal.
    //
    // Exists because the compressed and uncompressed paths pay their cost in
    // different places: uncompressed compares the query against a stored row
    // directly, while compressed does O(dimension) table work once and then
    // O(code_bytes) per node. Binding that setup to the query rather than
    // repeating it per distance is what keeps ADC cheaper than the fp32 scan,
    // and it keeps the branch out of the inner loop of search_layer().
    class Probe {
    public:
        Probe(const HierarchicalGraphIndex& owner, std::span<const float> query);
        // In the uncompressed case a Probe holds a span over the caller's
        // buffer, so a temporary would dangle for the rest of the traversal.
        // Deleting the rvalue overload turns that into a compile error rather
        // than a recall regression that only shows up in a benchmark.
        Probe(const HierarchicalGraphIndex&, std::vector<float>&&) = delete;
        [[nodiscard]] float to(NodeId node) const noexcept;

    private:
        const HierarchicalGraphIndex* owner_;
        std::span<const float> raw_;  // uncompressed: the query itself
        std::vector<float> lut_;      // compressed: the precomputed table
    };
    friend class Probe;

    // Row storage for one node: floats when uncompressed, code bytes otherwise.
    [[nodiscard]] std::span<const float> vector_of(NodeId node) const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> code_of(NodeId node) const noexcept;
    // Full-precision reconstruction of a node, for the paths that need a vector
    // rather than a distance (snapshot export, neighbor pruning).
    [[nodiscard]] std::vector<float> reconstruct(NodeId node) const;
    [[nodiscard]] bool compressed() const noexcept {
        return quantizer_ != nullptr;
    }

    [[nodiscard]] int random_level();
    // Beam search within one layer; returns up to `ef` nearest as (dist, node).
    [[nodiscard]] std::vector<Scored> search_layer(const Probe& probe,
                                                   NodeId entry, std::size_t ef,
                                                   int level) const;
    void connect(NodeId node, const std::vector<Scored>& candidates, int level,
                 std::size_t max_links);
    void insert_unchecked(const RecordID& id, std::span<const float> vector);

    Metric metric_;
    std::uint16_t dimension_;
    GraphParams params_;
    quant::QuantizerPtr quantizer_;
    std::size_t row_width_;
    double level_mult_;  // mL = 1 / ln(M)

    std::vector<float> data_;          // row-major fp32; empty when compressed
    std::vector<std::uint8_t> codes_;  // row-major codes; empty when not
    std::vector<RecordID> ids_;
    std::vector<bool> deleted_;
    std::vector<int> node_levels_;
    // links_[node][level] = neighbor node ids
    std::vector<std::vector<std::vector<NodeId>>> links_;
    std::unordered_map<RecordID, NodeId> id_to_node_;

    int entry_point_{-1};
    int max_level_{-1};
    std::size_t deleted_count_{0};
    mutable std::mt19937 rng_;
};

}  // namespace elips

#endif  // ELIPS_INDEX_ENGINE_HIERARCHICAL_GRAPH_INDEX_HPP
