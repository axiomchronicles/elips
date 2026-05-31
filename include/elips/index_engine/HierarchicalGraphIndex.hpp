#ifndef ELIPS_INDEX_ENGINE_HIERARCHICAL_GRAPH_INDEX_HPP
#define ELIPS_INDEX_ENGINE_HIERARCHICAL_GRAPH_INDEX_HPP

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/IndexPort.hpp"

namespace elips {

// Primary ANN index: a from-scratch Hierarchical Navigable Small World graph.
// Vectors are stored row-major; the graph is layered with probabilistic level
// assignment. Removal is a soft tombstone (graph navigation is preserved;
// deleted nodes are excluded from results), matching the MVCC delete model.
class HierarchicalGraphIndex final : public IndexPort {
public:
    HierarchicalGraphIndex(Metric metric, std::uint16_t dimension,
                           GraphParams params);

    void insert(const RecordID& id, std::span<const float> vector) override;
    void remove(const RecordID& id) override;
    [[nodiscard]] std::vector<Hit> search(std::span<const float> query,
                                          std::size_t k) const override;

    [[nodiscard]] std::size_t size() const noexcept override {
        return ids_.size() - deleted_count_;
    }
    [[nodiscard]] std::string_view type_name() const noexcept override {
        return "graph";
    }

private:
    using NodeId = std::uint32_t;
    using Scored = std::pair<float, NodeId>;  // (distance, node)

    [[nodiscard]] float distance_to(std::span<const float> query,
                                    NodeId node) const noexcept;
    [[nodiscard]] std::span<const float> vector_of(NodeId node) const noexcept;
    [[nodiscard]] int random_level();
    // Beam search within one layer; returns up to `ef` nearest as (dist, node).
    [[nodiscard]] std::vector<Scored> search_layer(std::span<const float> query,
                                                   NodeId entry, std::size_t ef,
                                                   int level) const;
    void connect(NodeId node, const std::vector<Scored>& candidates, int level,
                 std::size_t max_links);

    Metric metric_;
    std::uint16_t dimension_;
    GraphParams params_;
    double level_mult_;  // mL = 1 / ln(M)

    std::vector<float> data_;          // row-major, dimension_ floats per node
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
