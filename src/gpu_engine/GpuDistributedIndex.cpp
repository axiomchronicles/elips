#include "elips/gpu_engine/GpuDistributedIndex.hpp"

#include <algorithm>
#include <expected>

#include "elips/gpu_engine/GpuIndexTransferManager.hpp"

namespace elips::gpu {
namespace {

std::vector<std::vector<RecordID>> split_ids(std::span<const RecordID> ids,
                                             std::size_t shards) {
    std::vector<std::vector<RecordID>> out(shards);
    for (std::size_t shard = 0; shard < shards; ++shard) {
        const auto begin = ids.size() * shard / shards;
        const auto end = ids.size() * (shard + 1) / shards;
        out[shard].assign(ids.begin() + static_cast<std::ptrdiff_t>(begin),
                          ids.begin() + static_cast<std::ptrdiff_t>(end));
    }
    return out;
}

std::vector<std::vector<float>> split_vectors(std::span<const float> vectors,
                                              std::size_t dim,
                                              std::size_t shards) {
    const auto count = vectors.size() / dim;
    std::vector<std::vector<float>> out(shards);
    for (std::size_t shard = 0; shard < shards; ++shard) {
        const auto begin = count * shard / shards;
        const auto end = count * (shard + 1) / shards;
        out[shard].assign(
            vectors.begin() + static_cast<std::ptrdiff_t>(begin * dim),
            vectors.begin() + static_cast<std::ptrdiff_t>(end * dim));
    }
    return out;
}

}  // namespace

GpuDistributedIndex::GpuDistributedIndex(
    std::vector<std::unique_ptr<GpuIndexPort>> indexes, DistributedMode mode)
    : indexes_(std::move(indexes)), mode_(mode) {}

void GpuDistributedIndex::insert(const RecordID& id, std::span<const float> vector) {
    if (indexes_.empty()) {
        return;
    }

    if (mode_ == DistributedMode::replicate) {
        for (auto& index : indexes_) {
            index->insert(id, vector);
        }
        return;
    }

    const auto shard =
        next_replica_.fetch_add(1, std::memory_order_relaxed) % indexes_.size();
    indexes_[shard]->insert(id, vector);
}

void GpuDistributedIndex::remove(const RecordID& id) {
    for (auto& index : indexes_) {
        index->remove(id);
    }
}

std::vector<IndexPort::Hit> GpuDistributedIndex::merge_hits(
    const std::vector<std::vector<Hit>>& shard_hits, std::size_t k) const {
    std::vector<Hit> merged;
    for (const auto& hits : shard_hits) {
        merged.insert(merged.end(), hits.begin(), hits.end());
    }
    const auto take = std::min(k, merged.size());
    std::partial_sort(merged.begin(),
                      merged.begin() + static_cast<std::ptrdiff_t>(take),
                      merged.end(),
                      [](const Hit& lhs, const Hit& rhs) {
                          return lhs.second < rhs.second;
                      });
    merged.resize(take);
    return merged;
}

std::vector<IndexPort::Hit> GpuDistributedIndex::search(std::span<const float> query,
                                                         std::size_t k) const {
    if (indexes_.empty() || k == 0) {
        return {};
    }

    if (mode_ == DistributedMode::replicate) {
        const auto replica =
            next_replica_.fetch_add(1, std::memory_order_relaxed) % indexes_.size();
        return indexes_[replica]->search(query, k);
    }

    std::vector<std::vector<Hit>> shard_hits;
    shard_hits.reserve(indexes_.size());
    for (const auto& index : indexes_) {
        shard_hits.push_back(index->search(query, k));
    }
    return merge_hits(shard_hits, k);
}

std::size_t GpuDistributedIndex::size() const noexcept {
    std::size_t total = 0;
    for (const auto& index : indexes_) {
        total += index->size();
    }
    if (mode_ == DistributedMode::replicate && !indexes_.empty()) {
        return indexes_.front()->size();
    }
    return total;
}

std::string_view GpuDistributedIndex::type_name() const noexcept {
    return mode_ == DistributedMode::replicate ? "gpu_replicated" : "gpu_sharded";
}

std::expected<void, GpuError>
GpuDistributedIndex::build_from_batch(std::span<const float> vectors,
                                      std::span<const RecordID> ids,
                                      const GpuIndexBuildParams& params) {
    if (indexes_.empty()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }

    if (mode_ == DistributedMode::replicate) {
        for (auto& index : indexes_) {
            auto built = index->build_from_batch(vectors, ids, params);
            if (!built.has_value()) {
                return built;
            }
        }
        return {};
    }

    auto prototype = indexes_.front()->export_snapshot();
    if (!prototype.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    const auto shard_ids = split_ids(ids, indexes_.size());
    const auto shard_vectors =
        split_vectors(vectors, prototype->dimension, indexes_.size());

    for (std::size_t shard = 0; shard < indexes_.size(); ++shard) {
        auto built = indexes_[shard]->build_from_batch(shard_vectors[shard], shard_ids[shard], params);
        if (!built.has_value()) {
            return built;
        }
    }
    return {};
}

std::expected<std::vector<std::vector<SearchResult>>, GpuError>
GpuDistributedIndex::search_batch(std::span<const float> queries, size_t k,
                                  size_t ef_search) const {
    if (indexes_.empty()) {
        return std::vector<std::vector<SearchResult>>{};
    }

    auto prototype = indexes_.front()->export_snapshot();
    if (!prototype.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    if (prototype->dimension == 0 || queries.size() % prototype->dimension != 0) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }

    const auto nq = static_cast<unsigned>(queries.size() / prototype->dimension);
    std::vector<std::vector<SearchResult>> out(nq);

    if (mode_ == DistributedMode::replicate) {
        const auto replica =
            next_replica_.fetch_add(1, std::memory_order_relaxed) % indexes_.size();
        return indexes_[replica]->search_batch(queries, k, ef_search);
    }

    std::vector<std::expected<std::vector<std::vector<SearchResult>>, GpuError>> shard_results;
    shard_results.reserve(indexes_.size());
    for (const auto& index : indexes_) {
        shard_results.push_back(index->search_batch(queries, k, ef_search));
    }
    for (const auto& result : shard_results) {
        if (!result.has_value()) {
            return std::unexpected(result.error());
        }
    }

    for (std::size_t q = 0; q < nq; ++q) {
        std::vector<SearchResult> merged;
        for (const auto& result : shard_results) {
            merged.insert(merged.end(), result->at(q).begin(), result->at(q).end());
        }
        const auto take = std::min(k, merged.size());
        std::partial_sort(merged.begin(),
                          merged.begin() + static_cast<std::ptrdiff_t>(take),
                          merged.end(),
                          [](const SearchResult& lhs, const SearchResult& rhs) {
                              return lhs.distance < rhs.distance;
                          });
        merged.resize(take);
        out[q] = std::move(merged);
    }

    return out;
}

std::expected<void, GpuError>
GpuDistributedIndex::export_to_cpu_index(elips::IndexPort& cpu_index_out) const {
    auto* transfer = dynamic_cast<elips::IndexTransferPort*>(&cpu_index_out);
    if (!transfer) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    auto snapshot = export_snapshot();
    if (!snapshot.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    auto imported = transfer->import_snapshot(*snapshot);
    if (!imported.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return {};
}

std::expected<void, GpuError>
GpuDistributedIndex::import_from_cpu_index(const elips::IndexPort& cpu_index) {
    const auto* transfer =
        dynamic_cast<const elips::IndexTransferPort*>(&cpu_index);
    if (!transfer) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    auto snapshot = transfer->export_snapshot();
    if (!snapshot.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    auto imported = import_snapshot(*snapshot);
    if (!imported.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return {};
}

size_t GpuDistributedIndex::device_bytes_used() const noexcept {
    std::size_t total = 0;
    for (const auto& index : indexes_) {
        total += index->device_bytes_used();
    }
    return total;
}

std::string_view GpuDistributedIndex::backend_name() const noexcept {
    return backend_name_;
}

std::expected<elips::IndexSnapshot, std::string>
GpuDistributedIndex::export_snapshot() const {
    if (indexes_.empty()) {
        return std::unexpected("distributed index has no sub-indexes");
    }

    if (mode_ == DistributedMode::replicate) {
        auto snapshot = indexes_.front()->export_snapshot();
        if (snapshot.has_value()) {
            snapshot->kind = elips::IndexSnapshotKind::gpu_distributed;
        }
        return snapshot;
    }

    std::vector<RecordID> ids;
    std::vector<float> vectors;
    elips::Metric metric = elips::Metric::cosine;
    std::uint16_t dimension = 0;
    for (const auto& index : indexes_) {
        auto snapshot = index->export_snapshot();
        if (!snapshot.has_value()) {
            return snapshot;
        }
        metric = snapshot->metric;
        dimension = snapshot->dimension;
        ids.insert(ids.end(), snapshot->ids.begin(), snapshot->ids.end());
        vectors.insert(vectors.end(), snapshot->vectors.begin(), snapshot->vectors.end());
    }

    elips::IndexSnapshot merged;
    merged.kind = elips::IndexSnapshotKind::gpu_distributed;
    merged.metric = metric;
    merged.dimension = dimension;
    merged.ids = std::move(ids);
    merged.vectors = std::move(vectors);
    return merged;
}

std::expected<void, std::string>
GpuDistributedIndex::import_snapshot(const elips::IndexSnapshot& snapshot) {
    if (indexes_.empty()) {
        return std::unexpected("distributed index has no sub-indexes");
    }

    if (mode_ == DistributedMode::replicate) {
        for (auto& index : indexes_) {
            auto imported = index->import_snapshot(snapshot);
            if (!imported.has_value()) {
                return imported;
            }
        }
        return {};
    }

    const auto shard_ids = split_ids(snapshot.ids, indexes_.size());
    const auto shard_vectors =
        split_vectors(snapshot.vectors, snapshot.dimension, indexes_.size());

    for (std::size_t shard = 0; shard < indexes_.size(); ++shard) {
        elips::IndexSnapshot shard_snapshot = snapshot;
        shard_snapshot.ids = shard_ids[shard];
        shard_snapshot.vectors = shard_vectors[shard];
        shard_snapshot.ivf.reset();
        shard_snapshot.pq.reset();
        auto imported = indexes_[shard]->import_snapshot(shard_snapshot);
        if (!imported.has_value()) {
            return imported;
        }
    }
    return {};
}

}  // namespace elips::gpu
