#include "elips/gpu_engine/GpuIVFPQIndex.hpp"

#include <algorithm>
#include <expected>
#include <limits>
#include <numeric>

#include "detail/IvfIndexState.hpp"
#include "elips/domain/Errors.hpp"
#include "elips/gpu_engine/GpuIndexTransferManager.hpp"
#include "elips/gpu_engine/GpuQuantizationPipeline.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu {
namespace {

void validate_dimension(std::span<const float> vector, uint16_t dimension,
                        std::string_view label) {
    if (vector.size() != dimension) {
        throw elips::DimensionMismatch{std::string(label) +
                                       " dimension does not match GPU IVFPQ index"};
    }
}

uint32_t choose_pq_dim(uint16_t dimension, uint32_t configured) noexcept {
    if (configured > 0 && dimension % configured == 0) {
        return configured;
    }

    uint32_t guess = std::max<uint32_t>(1, dimension / 8);
    while (guess > 1 && dimension % guess != 0) {
        --guess;
    }
    return guess;
}

std::vector<IndexPort::Hit> exact_rank(GpuPort& backend, elips::Metric metric,
                                       uint16_t dimension,
                                       std::span<const float> query,
                                       const std::vector<RecordID>& ids,
                                       const std::vector<float>& vectors,
                                       std::size_t k) {
    if (ids.empty() || k == 0) {
        return {};
    }

    std::vector<float> distances(ids.size());
    auto dist = backend.compute_distances_batch(query, vectors, distances, 1, ids.size(),
                                                dimension, metric);
    if (!dist.has_value()) {
        return {};
    }

    const auto take = std::min(k, ids.size());
    std::vector<std::uint32_t> indices(take);
    std::vector<float> values(take);
    auto tk = backend.top_k(distances, indices, values, 1, ids.size(), take);
    if (!tk.has_value()) {
        return {};
    }

    std::vector<IndexPort::Hit> hits;
    hits.reserve(take);
    for (std::size_t i = 0; i < take; ++i) {
        if (indices[i] >= ids.size()) {
            continue;
        }
        hits.emplace_back(ids[indices[i]], values[i]);
    }
    return hits;
}

}  // namespace

GpuIVFPQIndex::GpuIVFPQIndex(GpuPort& backend, elips::Metric metric,
                             uint16_t dimension, const GpuConfig& config)
    : backend_(backend),
      metric_(metric),
      dimension_(dimension),
      config_(config),
      pq_dim_(choose_pq_dim(dimension, config.ivf_pq_params.pq_dim)),
      pq_bits_(std::clamp<std::uint32_t>(config.ivf_pq_params.pq_bits, 4U, 8U)),
      backend_name_(backend.device_info().backend),
      state_(std::make_unique<detail::IvfIndexState>(metric, dimension, config)) {}

GpuIVFPQIndex::~GpuIVFPQIndex() = default;

void GpuIVFPQIndex::rebuild_codes() {
    codebook_.clear();
    codes_.clear();

    if (state_->size() == 0 || !state_->trained()) {
        return;
    }

    const auto ksub = 1U << pq_bits_;
    const auto sub_dim = static_cast<std::size_t>(dimension_) / pq_dim_;
    codebook_.resize(static_cast<std::size_t>(pq_dim_) * ksub * sub_dim);

    std::vector<float> residuals(state_->host_vectors().size(), 0.0F);
    const auto& vectors = state_->host_vectors();
    const auto& centroids = state_->centroids();
    const auto& assignments = state_->assignments();

    for (std::size_t slot = 0; slot < state_->size(); ++slot) {
        const auto list = assignments[slot];
        const auto vector_offset = slot * static_cast<std::size_t>(dimension_);
        const auto centroid_offset = list * static_cast<std::size_t>(dimension_);
        for (std::size_t dim = 0; dim < dimension_; ++dim) {
            residuals[vector_offset + dim] =
                vectors[vector_offset + dim] - centroids[centroid_offset + dim];
        }
    }

    GpuQuantizationPipeline pipeline(backend_);
    auto trained = pipeline.train_pq_codebook(
        residuals, state_->size(), dimension_, pq_dim_, ksub, codebook_);
    if (!trained.has_value()) {
        codebook_.clear();
        return;
    }

    codes_.resize(state_->size() * static_cast<std::size_t>(pq_dim_));
    auto encoded = pipeline.encode_pq(residuals, codebook_, state_->size(),
                                      dimension_, pq_dim_, codes_);
    if (!encoded.has_value()) {
        codebook_.clear();
        codes_.clear();
    }
}

std::vector<float> GpuIVFPQIndex::decode_candidate_vector(
    std::span<const std::uint8_t> code, std::span<const float> centroid) const {
    const auto ksub = 1U << pq_bits_;
    const auto sub_dim = static_cast<std::size_t>(dimension_) / pq_dim_;

    std::vector<float> decoded(dimension_, 0.0F);
    for (std::size_t m = 0; m < pq_dim_; ++m) {
        const auto code_value = static_cast<std::size_t>(code[m]);
        const auto centroid_base =
            ((m * ksub) + code_value) * sub_dim;
        for (std::size_t d = 0; d < sub_dim; ++d) {
            decoded[m * sub_dim + d] =
                centroid[m * sub_dim + d] + codebook_[centroid_base + d];
        }
    }
    return decoded;
}

void GpuIVFPQIndex::insert(const RecordID& id, std::span<const float> vector) {
    validate_dimension(vector, dimension_, "vector");
    state_->insert(backend_, id, vector);
    rebuild_codes();
}

void GpuIVFPQIndex::remove(const RecordID& id) {
    state_->remove(backend_, id);
    rebuild_codes();
}

std::vector<IndexPort::Hit> GpuIVFPQIndex::search(std::span<const float> query,
                                                  std::size_t k) const {
    validate_dimension(query, dimension_, "query");
    if (state_->size() == 0 || k == 0) {
        return {};
    }

    detail::IvfIndexState::CandidateSet candidates;
    if (!state_->trained()) {
        candidates = state_->all_candidates();
    } else {
        auto probed =
            state_->probe_lists(const_cast<GpuPort&>(backend_), query, state_->nprobe());
        if (!probed.has_value()) {
            candidates = state_->all_candidates();
        } else {
            candidates = state_->gather_candidates(*probed);
        }
    }
    if (candidates.ids.empty()) {
        return {};
    }

    if (codebook_.empty() || codes_.size() != state_->size() * static_cast<std::size_t>(pq_dim_)) {
        return exact_rank(backend_, metric_, dimension_, query, candidates.ids,
                          candidates.vectors, k);
    }

    struct ApproxHit {
        float distance;
        std::uint32_t slot;
    };

    std::vector<ApproxHit> approx;
    approx.reserve(candidates.ids.size());
    for (std::size_t i = 0; i < candidates.ids.size(); ++i) {
        const auto slot = candidates.source_slots[i];
        const auto list = candidates.source_lists[i];
        const auto code = std::span<const std::uint8_t>{
            codes_.data() + slot * static_cast<std::size_t>(pq_dim_),
            static_cast<std::size_t>(pq_dim_)};
        const auto centroid = std::span<const float>{
            state_->centroids().data() +
                list * static_cast<std::size_t>(dimension_),
            static_cast<std::size_t>(dimension_)};
        const auto decoded = decode_candidate_vector(code, centroid);
        approx.push_back({elips::distance(metric_, query, decoded), slot});
    }

    const auto rerank_count =
        std::min<std::size_t>(approx.size(), std::max<std::size_t>(k * 4, k));
    std::partial_sort(
        approx.begin(),
        approx.begin() + static_cast<std::ptrdiff_t>(rerank_count),
        approx.end(),
        [](const ApproxHit& lhs, const ApproxHit& rhs) {
            return lhs.distance < rhs.distance;
        });

    std::vector<RecordID> rerank_ids;
    std::vector<float> rerank_vectors;
    rerank_ids.reserve(rerank_count);
    rerank_vectors.reserve(rerank_count * static_cast<std::size_t>(dimension_));
    for (std::size_t i = 0; i < rerank_count; ++i) {
        const auto slot = approx[i].slot;
        rerank_ids.push_back(state_->ids()[slot]);
        const auto offset = slot * static_cast<std::size_t>(dimension_);
        rerank_vectors.insert(
            rerank_vectors.end(),
            state_->host_vectors().begin() + static_cast<std::ptrdiff_t>(offset),
            state_->host_vectors().begin() +
                static_cast<std::ptrdiff_t>(offset + dimension_));
    }

    return exact_rank(backend_, metric_, dimension_, query, rerank_ids,
                      rerank_vectors, k);
}

std::size_t GpuIVFPQIndex::size() const noexcept { return state_->size(); }

std::string_view GpuIVFPQIndex::type_name() const noexcept {
    return "gpu_ivf_pq";
}

std::expected<void, GpuError>
GpuIVFPQIndex::build_from_batch(std::span<const float> vectors,
                                std::span<const RecordID> ids,
                                const GpuIndexBuildParams& params) {
    (void)params;
    auto built = state_->build_from_batch(backend_, vectors, ids);
    if (!built.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    rebuild_codes();
    return {};
}

std::expected<std::vector<std::vector<SearchResult>>, GpuError>
GpuIVFPQIndex::search_batch(std::span<const float> queries, size_t k,
                            size_t ef_search) const {
    (void)ef_search;
    if (queries.size() % dimension_ != 0) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }

    const auto nq = queries.size() / dimension_;
    std::vector<std::vector<SearchResult>> out(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        const auto hits = search(
            std::span<const float>{queries.data() + q * dimension_, dimension_}, k);
        auto& row = out[q];
        row.reserve(hits.size());
        for (const auto& [id, distance] : hits) {
            row.push_back(SearchResult{.id = id, .distance = distance});
        }
    }
    return out;
}

std::expected<void, GpuError>
GpuIVFPQIndex::export_to_cpu_index(elips::IndexPort& cpu_index_out) const {
    auto* transfer = dynamic_cast<elips::IndexTransferPort*>(&cpu_index_out);
    if (!transfer) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    auto cloned = GpuIndexTransferManager::clone_gpu_to_cpu(*this, *transfer);
    if (!cloned.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return {};
}

std::expected<void, GpuError>
GpuIVFPQIndex::import_from_cpu_index(const elips::IndexPort& cpu_index) {
    const auto* transfer =
        dynamic_cast<const elips::IndexTransferPort*>(&cpu_index);
    if (!transfer) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return GpuIndexTransferManager::clone_cpu_to_gpu(*transfer, *this);
}

size_t GpuIVFPQIndex::device_bytes_used() const noexcept {
    return state_->device_bytes_used() +
           (codebook_.size() * sizeof(float)) +
           (codes_.size() * sizeof(std::uint8_t));
}

std::string_view GpuIVFPQIndex::backend_name() const noexcept {
    return backend_name_;
}

std::expected<elips::IndexSnapshot, std::string>
GpuIVFPQIndex::export_snapshot() const {
    auto snapshot = state_->export_snapshot(elips::IndexSnapshotKind::gpu_ivf_pq);
    if (!codebook_.empty()) {
        elips::PqSnapshot pq;
        pq.pq_dim = pq_dim_;
        pq.pq_bits = pq_bits_;
        pq.codebook = codebook_;
        pq.codes = codes_;
        snapshot.pq = std::move(pq);
    }
    return snapshot;
}

std::expected<void, std::string>
GpuIVFPQIndex::import_snapshot(const elips::IndexSnapshot& snapshot) {
    if (snapshot.pq.has_value()) {
        pq_dim_ = snapshot.pq->pq_dim;
        pq_bits_ = snapshot.pq->pq_bits;
        codebook_ = snapshot.pq->codebook;
        codes_ = snapshot.pq->codes;
    } else {
        codebook_.clear();
        codes_.clear();
    }

    auto imported = state_->import_snapshot(backend_, snapshot);
    if (!imported.has_value()) {
        return imported;
    }

    if (codebook_.empty() || codes_.size() != state_->size() * static_cast<std::size_t>(pq_dim_)) {
        rebuild_codes();
    }
    return {};
}

}  // namespace elips::gpu
