#include "elips/gpu_engine/GpuIVFFlatIndex.hpp"

#include <algorithm>
#include <expected>

#include "detail/IvfIndexState.hpp"
#include "elips/domain/Errors.hpp"
#include "elips/gpu_engine/GpuIndexTransferManager.hpp"
#include "elips/gpu_engine/GpuSearchPipeline.hpp"

namespace elips::gpu {
namespace {

void validate_dimension(std::span<const float> vector, uint16_t dimension,
                        std::string_view label) {
    if (vector.size() != dimension) {
        throw elips::DimensionMismatch{std::string(label) +
                                       " dimension does not match GPU IVF flat index"};
    }
}

}  // namespace

GpuIVFFlatIndex::GpuIVFFlatIndex(GpuPort& backend, elips::Metric metric,
                                 uint16_t dimension, const GpuConfig& config)
    : backend_(backend),
      metric_(metric),
      dimension_(dimension),
      config_(config),
      backend_name_(backend.device_info().backend),
      state_(std::make_unique<detail::IvfIndexState>(metric, dimension, config)) {}

GpuIVFFlatIndex::~GpuIVFFlatIndex() = default;

void GpuIVFFlatIndex::insert(const RecordID& id, std::span<const float> vector) {
    validate_dimension(vector, dimension_, "vector");
    state_->insert(backend_, id, vector);
}

void GpuIVFFlatIndex::remove(const RecordID& id) {
    state_->remove(backend_, id);
}

std::vector<IndexPort::Hit> GpuIVFFlatIndex::rank_candidates(
    std::span<const float> query, std::size_t k,
    const std::vector<RecordID>& candidate_ids,
    const std::vector<float>& candidate_vectors) const {
    if (candidate_ids.empty() || k == 0) {
        return {};
    }

    std::vector<float> distances(candidate_ids.size());
    auto dist = backend_.compute_distances_batch(
        query,
        std::span<const float>{candidate_vectors.data(), candidate_vectors.size()},
        distances,
        1,
        candidate_ids.size(),
        dimension_,
        metric_);
    if (!dist.has_value()) {
        return {};
    }

    const auto take = std::min(k, candidate_ids.size());
    std::vector<std::uint32_t> indices(take);
    std::vector<float> values(take);
    auto tk = backend_.top_k(distances, indices, values, 1, candidate_ids.size(), take);
    if (!tk.has_value()) {
        return {};
    }

    std::vector<Hit> hits;
    hits.reserve(take);
    for (std::size_t i = 0; i < take; ++i) {
        if (indices[i] >= candidate_ids.size()) {
            continue;
        }
        hits.emplace_back(candidate_ids[indices[i]], values[i]);
    }
    return hits;
}

std::vector<IndexPort::Hit> GpuIVFFlatIndex::search(std::span<const float> query,
                                                     std::size_t k) const {
    validate_dimension(query, dimension_, "query");
    if (state_->size() == 0 || k == 0) {
        return {};
    }

    detail::IvfIndexState::CandidateSet candidates;
    if (!state_->trained()) {
        candidates = state_->all_candidates();
    } else {
        auto probed = state_->probe_lists(backend_, query, state_->nprobe());
        if (!probed.has_value()) {
            return {};
        }
        candidates = state_->gather_candidates(*probed);
    }

    return rank_candidates(query, k, candidates.ids, candidates.vectors);
}

std::size_t GpuIVFFlatIndex::size() const noexcept { return state_->size(); }

std::string_view GpuIVFFlatIndex::type_name() const noexcept {
    return "gpu_ivf_flat";
}

std::expected<void, GpuError>
GpuIVFFlatIndex::build_from_batch(std::span<const float> vectors,
                                  std::span<const RecordID> ids,
                                  const GpuIndexBuildParams& params) {
    (void)params;
    auto result = state_->build_from_batch(backend_, vectors, ids);
    if (!result.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return {};
}

std::expected<std::vector<std::vector<SearchResult>>, GpuError>
GpuIVFFlatIndex::search_batch(std::span<const float> queries, size_t k,
                              size_t ef_search) const {
    (void)ef_search;
    if (queries.size() % dimension_ != 0) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }

    const auto nq = queries.size() / dimension_;
    std::vector<std::vector<SearchResult>> results(nq);
    if (nq == 0 || k == 0 || state_->size() == 0) {
        return results;
    }

    std::vector<std::vector<std::uint32_t>> probed_lists;
    if (state_->trained()) {
        auto probed = state_->probe_lists_batch(backend_, queries, state_->nprobe());
        if (!probed.has_value()) {
            return std::unexpected(probed.error());
        }
        probed_lists = std::move(*probed);
    } else {
        probed_lists.assign(nq, {});
    }

    for (std::size_t q = 0; q < nq; ++q) {
        const auto query = std::span<const float>{
            queries.data() + q * dimension_, static_cast<std::size_t>(dimension_)};
        auto candidates = state_->trained()
                ? state_->gather_candidates(probed_lists[q])
                : state_->all_candidates();
        const auto hits = rank_candidates(query, k, candidates.ids, candidates.vectors);

        auto& row = results[q];
        row.reserve(hits.size());
        for (const auto& [id, distance] : hits) {
            row.push_back(SearchResult{.id = id, .distance = distance});
        }
    }

    return results;
}

std::expected<void, GpuError>
GpuIVFFlatIndex::export_to_cpu_index(elips::IndexPort& cpu_index_out) const {
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
GpuIVFFlatIndex::import_from_cpu_index(const elips::IndexPort& cpu_index) {
    const auto* transfer =
        dynamic_cast<const elips::IndexTransferPort*>(&cpu_index);
    if (!transfer) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return GpuIndexTransferManager::clone_cpu_to_gpu(*transfer, *this);
}

size_t GpuIVFFlatIndex::device_bytes_used() const noexcept {
    return state_->device_bytes_used();
}

std::string_view GpuIVFFlatIndex::backend_name() const noexcept {
    return backend_name_;
}

std::expected<elips::IndexSnapshot, std::string>
GpuIVFFlatIndex::export_snapshot() const {
    return state_->export_snapshot(elips::IndexSnapshotKind::gpu_ivf_flat);
}

std::expected<void, std::string>
GpuIVFFlatIndex::import_snapshot(const elips::IndexSnapshot& snapshot) {
    return state_->import_snapshot(backend_, snapshot);
}

}  // namespace elips::gpu
