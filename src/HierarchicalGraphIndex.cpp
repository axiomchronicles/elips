#include "elips/index_engine/HierarchicalGraphIndex.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

#include "elips/vector_engine/Metrics.hpp"

namespace elips {

HierarchicalGraphIndex::HierarchicalGraphIndex(Metric metric,
                                               std::uint16_t dimension,
                                               GraphParams params)
    : metric_(metric),
      dimension_(dimension),
      params_(params),
      level_mult_(1.0 / std::log(static_cast<double>(std::max<std::size_t>(
                            params.max_connections, 2)))),
      rng_(std::random_device{}()) {}

std::span<const float> HierarchicalGraphIndex::vector_of(
    NodeId node) const noexcept {
    return {data_.data() + static_cast<std::size_t>(node) * dimension_,
            dimension_};
}

float HierarchicalGraphIndex::distance_to(std::span<const float> query,
                                          NodeId node) const noexcept {
    return distance(metric_, query, vector_of(node));
}

int HierarchicalGraphIndex::random_level() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng_);
    if (u <= 0.0) {
        u = std::numeric_limits<double>::min();
    }
    return static_cast<int>(-std::log(u) * level_mult_);
}

std::vector<HierarchicalGraphIndex::Scored>
HierarchicalGraphIndex::search_layer(std::span<const float> query, NodeId entry,
                                     std::size_t ef, int level) const {
    std::unordered_set<NodeId> visited;
    visited.insert(entry);
    const float entry_dist = distance_to(query, entry);

    // Candidate frontier: min-heap on distance (closest first).
    using MinHeap =
        std::priority_queue<Scored, std::vector<Scored>, std::greater<>>;
    // Result set: max-heap on distance (farthest on top, easy to evict).
    using MaxHeap = std::priority_queue<Scored>;
    MinHeap candidates;
    MaxHeap result;
    candidates.emplace(entry_dist, entry);
    result.emplace(entry_dist, entry);

    while (!candidates.empty()) {
        const auto [cur_dist, cur] = candidates.top();
        candidates.pop();
        if (cur_dist > result.top().first && result.size() >= ef) {
            break;
        }
        for (const NodeId neighbor : links_[cur][static_cast<std::size_t>(level)]) {
            if (!visited.insert(neighbor).second) {
                continue;
            }
            const float d = distance_to(query, neighbor);
            if (d < result.top().first || result.size() < ef) {
                candidates.emplace(d, neighbor);
                result.emplace(d, neighbor);
                if (result.size() > ef) {
                    result.pop();
                }
            }
        }
    }

    std::vector<Scored> out;
    out.reserve(result.size());
    while (!result.empty()) {
        out.push_back(result.top());
        result.pop();
    }
    std::sort(out.begin(), out.end());  // ascending by distance
    return out;
}

void HierarchicalGraphIndex::connect(NodeId node,
                                     const std::vector<Scored>& candidates,
                                     int level, std::size_t max_links) {
    // Diversity heuristic (HNSW Algorithm 4): keep a candidate only if it is
    // closer to the new node than to any already-selected neighbor. This yields
    // better-connected, higher-recall graphs than taking the closest M.
    std::vector<NodeId> selected;
    for (const auto& [dist_to_node, candidate] : candidates) {
        if (candidate == node) {
            continue;
        }
        if (selected.size() >= max_links) {
            break;
        }
        bool keep = true;
        for (const NodeId chosen : selected) {
            if (distance_to(vector_of(candidate), chosen) < dist_to_node) {
                keep = false;
                break;
            }
        }
        if (keep) {
            selected.push_back(candidate);
        }
    }

    auto& node_links = links_[node][static_cast<std::size_t>(level)];
    for (const NodeId neighbor : selected) {
        node_links.push_back(neighbor);

        // Add the reverse edge, pruning the neighbor to its closest max_links.
        auto& other_links = links_[neighbor][static_cast<std::size_t>(level)];
        other_links.push_back(node);
        if (other_links.size() > max_links) {
            const std::span<const float> base = vector_of(neighbor);
            std::sort(other_links.begin(), other_links.end(),
                      [&](NodeId a, NodeId b) {
                          return distance_to(base, a) < distance_to(base, b);
                      });
            other_links.resize(max_links);
        }
    }
}

void HierarchicalGraphIndex::insert(const RecordID& id,
                                    std::span<const float> vector) {
    const auto node = static_cast<NodeId>(ids_.size());
    const int level = random_level();
    ids_.push_back(id);
    deleted_.push_back(false);
    node_levels_.push_back(level);
    data_.insert(data_.end(), vector.begin(), vector.end());
    links_.emplace_back(static_cast<std::size_t>(level) + 1);
    id_to_node_[id] = node;

    if (entry_point_ < 0) {
        entry_point_ = static_cast<int>(node);
        max_level_ = level;
        return;
    }

    NodeId cur = static_cast<NodeId>(entry_point_);
    // Greedy descent through layers above the new node's top level.
    for (int l = max_level_; l > level; --l) {
        const auto found = search_layer(vector, cur, 1, l);
        if (!found.empty()) {
            cur = found.front().second;
        }
    }

    const std::size_t max0 = params_.max_connections * 2;
    for (int l = std::min(level, max_level_); l >= 0; --l) {
        const auto found = search_layer(vector, cur, params_.ef_construction, l);
        const std::size_t max_links =
            (l == 0) ? max0 : params_.max_connections;
        connect(node, found, l, max_links);
        if (!found.empty()) {
            cur = found.front().second;
        }
    }

    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = static_cast<int>(node);
    }
}

void HierarchicalGraphIndex::remove(const RecordID& id) {
    const auto it = id_to_node_.find(id);
    if (it == id_to_node_.end() || deleted_[it->second]) {
        return;
    }
    deleted_[it->second] = true;
    ++deleted_count_;
}

std::vector<IndexPort::Hit> HierarchicalGraphIndex::search(
    std::span<const float> query, std::size_t k) const {
    if (entry_point_ < 0 || k == 0) {
        return {};
    }
    NodeId cur = static_cast<NodeId>(entry_point_);
    for (int l = max_level_; l > 0; --l) {
        const auto found = search_layer(query, cur, 1, l);
        if (!found.empty()) {
            cur = found.front().second;
        }
    }
    const std::size_t ef = std::max(params_.ef_search, k);
    const auto found = search_layer(query, cur, ef, 0);

    std::vector<Hit> hits;
    hits.reserve(std::min(k, found.size()));
    for (const auto& [dist, node] : found) {
        if (deleted_[node]) {
            continue;
        }
        hits.emplace_back(ids_[node], dist);
        if (hits.size() >= k) {
            break;
        }
    }
    return hits;
}

}  // namespace elips
