#ifndef ELIPS_INDEX_ENGINE_INDEX_PORT_HPP
#define ELIPS_INDEX_ENGINE_INDEX_PORT_HPP

#include <cstddef>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "elips/domain/RecordID.hpp"

namespace elips {

// Port for pluggable index implementations (DIP). ExactIndex ships in v1.0;
// HierarchicalGraphIndex (HNSW) plugs in behind the same interface later.
class IndexPort {
public:
    using Hit = std::pair<RecordID, float>;  // (id, ordering-normalized distance)

    IndexPort() = default;
    virtual ~IndexPort() = default;
    IndexPort(const IndexPort&) = delete;
    IndexPort& operator=(const IndexPort&) = delete;
    IndexPort(IndexPort&&) = delete;
    IndexPort& operator=(IndexPort&&) = delete;

    // Write path. Vectors are pre-validated/normalized by the vector engine.
    virtual void insert(const RecordID& id, std::span<const float> vector) = 0;
    virtual void remove(const RecordID& id) = 0;

    // Search path: up to k nearest, sorted ascending by distance.
    [[nodiscard]] virtual std::vector<Hit> search(std::span<const float> query,
                                                  std::size_t k) const = 0;

    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual std::string_view type_name() const noexcept = 0;
};

}  // namespace elips

#endif  // ELIPS_INDEX_ENGINE_INDEX_PORT_HPP
