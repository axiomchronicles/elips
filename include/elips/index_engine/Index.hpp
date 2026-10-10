#ifndef ELIPS_INDEX_ENGINE_INDEX_HPP
#define ELIPS_INDEX_ENGINE_INDEX_HPP

#include <cstddef>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "elips/domain/RecordID.hpp"

namespace elips {

// Port for pluggable index implementations (DIP). IndexFlat ships in v1.0;
// IndexHNSW (HNSW) plugs in behind the same interface later.
class Index {
public:
    using Hit = std::pair<RecordID, float>;  // (id, ordering-normalized distance)

    Index() = default;
    virtual ~Index() = default;
    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;
    Index(Index&&) = delete;
    Index& operator=(Index&&) = delete;

    // Train path (FAISS-aligned). Flat/graph indexes need no training and keep
    // the no-op defaults; IVF-family indexes learn coarse centroids here before
    // any add_with_ids. `training_vectors` holds n * dimension row-major floats.
    virtual void train(std::span<const float> training_vectors, std::size_t n) {
        (void)training_vectors;
        (void)n;
    }
    [[nodiscard]] virtual bool is_trained() const noexcept { return true; }

    // Drop all stored records (training/centroids may be retained; see override).
    virtual void reset() {}

    // Write path. Vectors are pre-validated/normalized by the vector engine.
    virtual void add_with_ids(const RecordID& id, std::span<const float> vector) = 0;
    virtual void remove(const RecordID& id) = 0;

    // Search path: up to k nearest, sorted ascending by distance.
    [[nodiscard]] virtual std::vector<Hit> search(std::span<const float> query,
                                                  std::size_t k) const = 0;

    [[nodiscard]] virtual std::size_t ntotal() const noexcept = 0;
    [[nodiscard]] virtual std::string_view type_name() const noexcept = 0;

    // Reclaim space held by removed records. Indexes that delete eagerly have
    // nothing to do; tombstoning indexes (the HNSW graph) rebuild here.
    virtual void vacuum() {}

    // Records tombstoned but not yet reclaimed. Zero for eager-delete indexes.
    [[nodiscard]] virtual std::size_t pending_removals() const noexcept {
        return 0;
    }
};

using IndexPort [[deprecated("renamed to Index")]] = Index;

}  // namespace elips

#endif  // ELIPS_INDEX_ENGINE_INDEX_HPP
