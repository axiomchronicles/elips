#include "elips/index_engine/IndexIVFFlat.hpp"

#include <algorithm>
#include <limits>
#include <string>

#include "elips/domain/Errors.hpp"
#include "elips/quant_engine/detail/KMeans.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips {
namespace {

// k-means refinement passes for the coarse quantizer (FAISS clustering default
// is 25; the trainer stops early once centroids stop moving).
constexpr std::size_t kTrainIters = 25;

void validate_dimension(std::span<const float> vector, std::uint16_t dimension,
                        std::string_view label) {
    if (vector.size() != dimension) {
        throw DimensionMismatch{std::string(label) + ": expected d=" +
                                std::to_string(dimension) + ", got " +
                                std::to_string(vector.size())};
    }
}

}  // namespace

IndexIVFFlat::IndexIVFFlat(Metric metric, std::uint16_t dimension,
                           std::size_t nlist)
    : metric_(metric),
      dimension_(dimension),
      nlist_(std::max<std::size_t>(1, nlist)) {}

void IndexIVFFlat::train(std::span<const float> training_vectors,
                         std::size_t n) {
    if (n == 0 || training_vectors.size() != n * dimension_) {
        throw InvalidVector{"IVFFlat training set is empty or not row-major"};
    }
    centroids_.assign(nlist_ * dimension_, 0.0F);
    quant::detail::kmeans(training_vectors, n, dimension_, nlist_, kTrainIters,
                          centroids_);
    // (Re)training invalidates any existing postings; start the lists empty.
    lists_.assign(nlist_, InvList{});
    ntotal_ = 0;
    trained_ = true;
}

std::size_t IndexIVFFlat::assign(std::span<const float> vector) const noexcept {
    float best = std::numeric_limits<float>::max();
    std::size_t best_list = 0;
    for (std::size_t c = 0; c < nlist_; ++c) {
        const float d = distance(metric_, vector, centroid(c));
        if (d < best) {
            best = d;
            best_list = c;
        }
    }
    return best_list;
}

void IndexIVFFlat::add_with_ids(const RecordID& id,
                                std::span<const float> vector) {
    if (!trained_) {
        throw NotTrainedError{"IndexIVFFlat::add_with_ids before train()"};
    }
    validate_dimension(vector, dimension_, "vector");
    InvList& list = lists_[assign(vector)];
    list.ids.push_back(id);
    list.data.insert(list.data.end(), vector.begin(), vector.end());
    ++ntotal_;
}

void IndexIVFFlat::remove(const RecordID& id) {
    for (InvList& list : lists_) {
        const auto it = std::find(list.ids.begin(), list.ids.end(), id);
        if (it == list.ids.end()) {
            continue;
        }
        const auto row = static_cast<std::size_t>(it - list.ids.begin());
        list.ids.erase(it);
        const auto first =
            list.data.begin() + static_cast<std::ptrdiff_t>(row * dimension_);
        list.data.erase(first, first + dimension_);
        --ntotal_;
        return;  // ids are unique across the whole index
    }
}

std::vector<Index::Hit> IndexIVFFlat::search(std::span<const float> query,
                                             std::size_t k) const {
    validate_dimension(query, dimension_, "query");
    if (!trained_) {
        throw NotTrainedError{"IndexIVFFlat::search before train()"};
    }
    if (ntotal_ == 0 || k == 0) {
        return {};
    }

    // Rank centroids by distance to the query, then probe the nearest nprobe.
    std::vector<std::pair<float, std::size_t>> coarse;
    coarse.reserve(nlist_);
    for (std::size_t c = 0; c < nlist_; ++c) {
        coarse.emplace_back(distance(metric_, query, centroid(c)), c);
    }
    const std::size_t probe = std::min(std::max<std::size_t>(1, nprobe), nlist_);
    std::partial_sort(
        coarse.begin(), coarse.begin() + static_cast<std::ptrdiff_t>(probe),
        coarse.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    std::vector<Hit> scored;
    for (std::size_t p = 0; p < probe; ++p) {
        const InvList& list = lists_[coarse[p].second];
        for (std::size_t i = 0; i < list.ids.size(); ++i) {
            const std::span<const float> row{list.data.data() + (i * dimension_),
                                             dimension_};
            scored.emplace_back(list.ids[i], distance(metric_, query, row));
        }
    }

    const std::size_t take = std::min(k, scored.size());
    std::partial_sort(
        scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(take),
        scored.end(),
        [](const Hit& lhs, const Hit& rhs) { return lhs.second < rhs.second; });
    scored.resize(take);
    return scored;
}

std::vector<Index::Hit> IndexIVFFlat::range_search(std::span<const float> query,
                                                   float radius) const {
    validate_dimension(query, dimension_, "query");
    if (!trained_) {
        throw NotTrainedError{"IndexIVFFlat::range_search before train()"};
    }

    std::vector<Hit> hits;
    if (ntotal_ == 0) {
        return hits;
    }

    // Same coarse-probe set as search(): rank centroids, probe the nearest
    // nprobe lists, keep everything within radius.
    std::vector<std::pair<float, std::size_t>> coarse;
    coarse.reserve(nlist_);
    for (std::size_t c = 0; c < nlist_; ++c) {
        coarse.emplace_back(distance(metric_, query, centroid(c)), c);
    }
    const std::size_t probe = std::min(std::max<std::size_t>(1, nprobe), nlist_);
    std::partial_sort(
        coarse.begin(), coarse.begin() + static_cast<std::ptrdiff_t>(probe),
        coarse.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    for (std::size_t p = 0; p < probe; ++p) {
        const InvList& list = lists_[coarse[p].second];
        for (std::size_t i = 0; i < list.ids.size(); ++i) {
            const std::span<const float> row{list.data.data() + (i * dimension_),
                                             dimension_};
            const float d = distance(metric_, query, row);
            if (d <= radius) {
                hits.emplace_back(list.ids[i], d);
            }
        }
    }

    std::sort(hits.begin(), hits.end(),
              [](const Hit& lhs, const Hit& rhs) { return lhs.second < rhs.second; });
    return hits;
}

void IndexIVFFlat::reset() {
    // FAISS semantics: drop postings, keep the trained coarse quantizer.
    for (InvList& list : lists_) {
        list.ids.clear();
        list.data.clear();
    }
    ntotal_ = 0;
}

}  // namespace elips
