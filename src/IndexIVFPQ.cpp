#include "elips/index_engine/IndexIVFPQ.hpp"

#include <algorithm>
#include <limits>
#include <string>

#include "elips/domain/Errors.hpp"
#include "elips/quant_engine/detail/KMeans.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips {
namespace {

// k-means passes for the coarse quantizer and the PQ codebooks (FAISS
// clustering default 25; the trainers stop early once centroids settle).
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

IndexIVFPQ::IndexIVFPQ(Metric metric, std::uint16_t dimension, std::size_t nlist,
                       std::uint32_t pq_dim, std::uint32_t pq_bits)
    : metric_(metric),
      dimension_(dimension),
      nlist_(std::max<std::size_t>(1, nlist)),
      pq_dim_(pq_dim),
      pq_bits_(pq_bits),
      code_bytes_(pq_dim) {
    if (pq_dim == 0 || dimension % pq_dim != 0) {
        throw ConfigError{"IndexIVFPQ: pq_dim must be a nonzero divisor of the "
                          "dimension"};
    }
    if (pq_bits < 4 || pq_bits > 8) {
        throw ConfigError{"IndexIVFPQ: pq_bits must be in [4, 8]"};
    }
}

void IndexIVFPQ::train(std::span<const float> training_vectors, std::size_t n) {
    if (n == 0 || training_vectors.size() != n * dimension_) {
        throw InvalidVector{"IVFPQ training set is empty or not row-major"};
    }

    // Stage 1: coarse quantizer.
    centroids_.assign(nlist_ * dimension_, 0.0F);
    quant::detail::kmeans(training_vectors, n, dimension_, nlist_, kTrainIters,
                          centroids_);

    // Stage 2: PQ on residuals (FAISS by_residual=true). Each training vector is
    // reduced to vector - centroid[assign], and the codebooks are fit to those.
    std::vector<float> residuals(n * dimension_);
    for (std::size_t row = 0; row < n; ++row) {
        const std::span<const float> v{
            training_vectors.data() + (row * dimension_), dimension_};
        const std::span<const float> c = centroid(assign(v));
        float* out = residuals.data() + (row * dimension_);
        for (std::size_t i = 0; i < dimension_; ++i) {
            out[i] = v[i] - c[i];
        }
    }
    pq_ = quant::ProductQuantizer::train(metric_, dimension_, pq_dim_, pq_bits_,
                                         kTrainIters, /*opq_iters=*/0, residuals,
                                         n);

    lists_.assign(nlist_, InvList{});
    ntotal_ = 0;
    trained_ = true;
}

std::size_t IndexIVFPQ::assign(std::span<const float> vector) const noexcept {
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

void IndexIVFPQ::add_with_ids(const RecordID& id,
                              std::span<const float> vector) {
    if (!trained_) {
        throw NotTrainedError{"IndexIVFPQ::add_with_ids before train()"};
    }
    validate_dimension(vector, dimension_, "vector");

    const std::size_t list_idx = assign(vector);
    const std::span<const float> c = centroid(list_idx);
    std::vector<float> residual(dimension_);
    for (std::size_t i = 0; i < dimension_; ++i) {
        residual[i] = vector[i] - c[i];
    }

    InvList& list = lists_[list_idx];
    const std::size_t off = list.codes.size();
    list.codes.resize(off + code_bytes_);
    pq_->encode(residual, std::span<std::uint8_t>{list.codes}.subspan(off,
                                                                      code_bytes_));
    list.ids.push_back(id);
    ++ntotal_;
}

void IndexIVFPQ::remove(const RecordID& id) {
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
        return;  // ids are unique across the whole index
    }
}

std::vector<Index::Hit> IndexIVFPQ::search(std::span<const float> query,
                                           std::size_t k) const {
    validate_dimension(query, dimension_, "query");
    if (!trained_) {
        throw NotTrainedError{"IndexIVFPQ::search before train()"};
    }
    if (ntotal_ == 0 || k == 0) {
        return {};
    }

    // Rank centroids, probe the nearest nprobe.
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

    // Metric split for residual ADC (see header + the derivation below):
    //   euclidean: ||q - x|| = ||(q-c) - r||, so the LUT is built from the query
    //     residual (q - centroid), fresh per probed list, and scores directly.
    //   dot/cosine: dot(q, x) = dot(q, c) + dot(q, r); the LUT built from the raw
    //     query scores dot(q, r), and the per-list constant dot(q, c) is added
    //     back. distance(dot_product, q, c) is exactly -dot(q, c), the offset.
    const bool residual_lut = metric_ == Metric::euclidean;
    std::vector<float> shared_lut;
    if (!residual_lut) {
        shared_lut = pq_->make_lut(query);
    }

    std::vector<Hit> scored;
    scored.reserve(ntotal_);
    std::vector<float> residual(dimension_);
    for (std::size_t p = 0; p < probe; ++p) {
        const std::size_t list_idx = coarse[p].second;
        const InvList& list = lists_[list_idx];
        const std::span<const float> c = centroid(list_idx);

        std::span<const float> lut;
        std::vector<float> residual_lut_storage;
        float offset = 0.0F;
        if (residual_lut) {
            for (std::size_t i = 0; i < dimension_; ++i) {
                residual[i] = query[i] - c[i];
            }
            residual_lut_storage = pq_->make_lut(residual);
            lut = residual_lut_storage;
        } else {
            lut = shared_lut;
            offset = distance(Metric::dot_product, query, c);  // == -dot(q, c)
        }

        for (std::size_t i = 0; i < list.ids.size(); ++i) {
            const std::span<const std::uint8_t> code{
                list.codes.data() + (i * code_bytes_), code_bytes_};
            scored.emplace_back(list.ids[i],
                                pq_->lut_distance(lut, code) + offset);
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

void IndexIVFPQ::reset() {
    // FAISS semantics: drop postings, keep the trained coarse quantizer + PQ.
    for (InvList& list : lists_) {
        list.ids.clear();
        list.codes.clear();
    }
    ntotal_ = 0;
}

}  // namespace elips
