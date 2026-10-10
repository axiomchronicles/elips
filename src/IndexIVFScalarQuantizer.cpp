#include "elips/index_engine/IndexIVFScalarQuantizer.hpp"

#include <algorithm>
#include <limits>
#include <string>

#include "elips/domain/Errors.hpp"
#include "elips/quant_engine/detail/KMeans.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips {
namespace {

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

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer(Metric metric,
                                                 std::uint16_t dimension,
                                                 std::size_t nlist)
    : metric_(metric),
      dimension_(dimension),
      nlist_(std::max<std::size_t>(1, nlist)),
      code_bytes_(dimension) {}

void IndexIVFScalarQuantizer::train(std::span<const float> training_vectors,
                                    std::size_t n) {
    if (n == 0 || training_vectors.size() != n * dimension_) {
        throw InvalidVector{"IVF-SQ training set is empty or not row-major"};
    }

    centroids_.assign(nlist_ * dimension_, 0.0F);
    quant::detail::kmeans(training_vectors, n, dimension_, nlist_, kTrainIters,
                          centroids_);

    // One global SQ codec over the raw training vectors (see header: SQ is
    // near-lossless per dimension, so residual coding is not worth a per-list
    // codec).
    sq_ = quant::ScalarQuantizer::train(metric_, dimension_, training_vectors, n);

    lists_.assign(nlist_, InvList{});
    ntotal_ = 0;
    trained_ = true;
}

std::size_t IndexIVFScalarQuantizer::assign(
    std::span<const float> vector) const noexcept {
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

void IndexIVFScalarQuantizer::add_with_ids(const RecordID& id,
                                           std::span<const float> vector) {
    if (!trained_) {
        throw NotTrainedError{"IndexIVFScalarQuantizer::add_with_ids before train()"};
    }
    validate_dimension(vector, dimension_, "vector");

    InvList& list = lists_[assign(vector)];
    const std::size_t off = list.codes.size();
    list.codes.resize(off + code_bytes_);
    sq_->encode(vector, std::span<std::uint8_t>{list.codes}.subspan(off,
                                                                    code_bytes_));
    list.ids.push_back(id);
    ++ntotal_;
}

void IndexIVFScalarQuantizer::remove(const RecordID& id) {
    for (InvList& list : lists_) {
        const auto it = std::find(list.ids.begin(), list.ids.end(), id);
        if (it == list.ids.end()) {
            continue;
        }
        const auto row = static_cast<std::size_t>(it - list.ids.begin());
        list.ids.erase(it);
        const auto first = list.codes.begin() +
                           static_cast<std::ptrdiff_t>(row * code_bytes_);
        list.codes.erase(first,
                         first + static_cast<std::ptrdiff_t>(code_bytes_));
        --ntotal_;
        return;
    }
}

std::vector<Index::Hit> IndexIVFScalarQuantizer::search(
    std::span<const float> query, std::size_t k) const {
    validate_dimension(query, dimension_, "query");
    if (!trained_) {
        throw NotTrainedError{"IndexIVFScalarQuantizer::search before train()"};
    }
    if (ntotal_ == 0 || k == 0) {
        return {};
    }

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

    // Codes encode the raw vector, so the ADC LUT is query-only and shared
    // across all probed lists.
    const std::vector<float> lut = sq_->make_lut(query);

    std::vector<Hit> scored;
    scored.reserve(ntotal_);
    for (std::size_t p = 0; p < probe; ++p) {
        const InvList& list = lists_[coarse[p].second];
        for (std::size_t i = 0; i < list.ids.size(); ++i) {
            const std::span<const std::uint8_t> code{
                list.codes.data() + (i * code_bytes_), code_bytes_};
            scored.emplace_back(list.ids[i], sq_->lut_distance(lut, code));
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

void IndexIVFScalarQuantizer::reset() {
    for (InvList& list : lists_) {
        list.ids.clear();
        list.codes.clear();
    }
    ntotal_ = 0;
}

}  // namespace elips
