#include "elips/gpu_engine/GpuGraphIndex.hpp"

#include <algorithm>
#include <expected>
#include <span>

#include "elips/domain/Errors.hpp"
#include "elips/gpu_engine/GpuIndexTransferManager.hpp"
#include "elips/index_engine/HierarchicalGraphIndex.hpp"

namespace elips::gpu {
namespace {

elips::GraphParams to_cpu_graph_params(const GpuConfig& config) {
    elips::GraphParams params;
    params.max_connections = std::max<std::size_t>(config.graph_params.graph_degree, 2);
    params.ef_construction = std::max<std::size_t>(
        config.graph_params.intermediate_graph_degree, params.max_connections);
    params.ef_search = std::max<std::size_t>(config.default_ef_search_gpu, 1);
    return params;
}

void validate_dimension(std::span<const float> vector, uint16_t dimension,
                        std::string_view label) {
    if (vector.size() != dimension) {
        throw elips::DimensionMismatch{std::string(label) +
                                       " dimension does not match GPU graph index"};
    }
}

}  // namespace

GpuGraphIndex::GpuGraphIndex(GpuPort& backend, elips::Metric metric,
                             uint16_t dimension, const GpuConfig& config)
    : backend_(backend),
      metric_(metric),
      dimension_(dimension),
      cpu_graph_(std::make_unique<elips::HierarchicalGraphIndex>(
          metric, dimension, to_cpu_graph_params(config))),
      backend_name_(backend.device_info().backend),
      config_(config) {}

GpuGraphIndex::~GpuGraphIndex() { release_graph_data(); }

void GpuGraphIndex::release_graph_data() noexcept {
    if (!graph_data_) {
        return;
    }
    backend_.free_device(std::move(graph_data_));
}

void GpuGraphIndex::sync_device_data_best_effort() noexcept {
    release_graph_data();
    if (host_vectors_.empty()) {
        return;
    }

    auto alloc = backend_.allocate_device(host_vectors_.size() * sizeof(float));
    if (!alloc.has_value()) {
        return;
    }

    graph_data_ = std::move(*alloc);
    auto upload = backend_.upload(host_vectors_.data(), graph_data_,
                                  host_vectors_.size() * sizeof(float));
    if (!upload.has_value()) {
        release_graph_data();
    }
}

void GpuGraphIndex::insert(const RecordID& id, std::span<const float> vector) {
    validate_dimension(vector, dimension_, "vector");
    if (id_to_slot_.contains(id)) {
        remove(id);
    }

    cpu_graph_->insert(id, vector);
    ids_.push_back(id);
    host_vectors_.insert(host_vectors_.end(), vector.begin(), vector.end());
    id_to_slot_[id] = ids_.size() - 1;
    count_ = ids_.size();
    sync_device_data_best_effort();
}

void GpuGraphIndex::remove(const RecordID& id) {
    const auto found = id_to_slot_.find(id);
    if (found == id_to_slot_.end()) {
        return;
    }

    cpu_graph_->remove(id);
    const auto slot = found->second;
    const auto last = ids_.size() - 1;
    const auto slot_begin =
        static_cast<std::ptrdiff_t>(slot * static_cast<std::size_t>(dimension_));
    const auto last_begin =
        static_cast<std::ptrdiff_t>(last * static_cast<std::size_t>(dimension_));

    if (slot != last) {
        ids_[slot] = ids_[last];
        std::copy(host_vectors_.begin() + last_begin,
                  host_vectors_.begin() + last_begin + dimension_,
                  host_vectors_.begin() + slot_begin);
        id_to_slot_[ids_[slot]] = slot;
    }

    ids_.pop_back();
    host_vectors_.erase(host_vectors_.begin() + last_begin,
                        host_vectors_.begin() + last_begin + dimension_);
    id_to_slot_.erase(found);
    count_ = ids_.size();
    sync_device_data_best_effort();
}

std::vector<IndexPort::Hit> GpuGraphIndex::search(std::span<const float> query,
                                                   std::size_t k) const {
    validate_dimension(query, dimension_, "query");
    return cpu_graph_->search(query, k);
}

std::size_t GpuGraphIndex::size() const noexcept { return count_; }
std::string_view GpuGraphIndex::type_name() const noexcept { return "gpu_graph"; }

std::expected<void, GpuError>
GpuGraphIndex::build_from_batch(std::span<const float> vectors,
                                std::span<const RecordID> ids,
                                const GpuIndexBuildParams& params) {
    auto* graph_params = std::get_if<GraphIndexBuildParams>(&params.params);
    if (!graph_params) return std::unexpected(GpuError::IndexBuildFailed);
    config_.graph_params = *graph_params;

    elips::IndexSnapshot snapshot;
    snapshot.kind = elips::IndexSnapshotKind::gpu_graph;
    snapshot.metric = metric_;
    snapshot.dimension = dimension_;
    snapshot.ids.assign(ids.begin(), ids.end());
    snapshot.vectors.assign(vectors.begin(), vectors.end());
    auto imported = import_snapshot(snapshot);
    if (!imported.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return {};
}

std::expected<std::vector<std::vector<SearchResult>>, GpuError>
GpuGraphIndex::search_batch(std::span<const float> queries, size_t k,
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
GpuGraphIndex::export_to_cpu_index(elips::IndexPort& cpu_index_out) const {
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
GpuGraphIndex::import_from_cpu_index(const elips::IndexPort& cpu_index) {
    const auto* transfer =
        dynamic_cast<const elips::IndexTransferPort*>(&cpu_index);
    if (!transfer) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return GpuIndexTransferManager::clone_cpu_to_gpu(*transfer, *this);
}

size_t GpuGraphIndex::device_bytes_used() const noexcept {
    return graph_data_.bytes();
}

std::string_view GpuGraphIndex::backend_name() const noexcept {
    return backend_name_;
}

std::expected<elips::IndexSnapshot, std::string>
GpuGraphIndex::export_snapshot() const {
    elips::IndexSnapshot snapshot;
    snapshot.kind = elips::IndexSnapshotKind::gpu_graph;
    snapshot.metric = metric_;
    snapshot.dimension = dimension_;
    snapshot.ids = ids_;
    snapshot.vectors = host_vectors_;
    return snapshot;
}

std::expected<void, std::string>
GpuGraphIndex::import_snapshot(const elips::IndexSnapshot& snapshot) {
    if (snapshot.dimension != dimension_) {
        return std::unexpected("snapshot dimension does not match GpuGraphIndex");
    }
    if (snapshot.metric != metric_) {
        return std::unexpected("snapshot metric does not match GpuGraphIndex");
    }
    if (snapshot.vectors.size() !=
        snapshot.ids.size() * static_cast<std::size_t>(dimension_)) {
        return std::unexpected(
            "snapshot vector payload is not row-major graph index data");
    }

    auto cpu_import = cpu_graph_->import_snapshot(snapshot);
    if (!cpu_import.has_value()) {
        return std::unexpected(cpu_import.error());
    }

    ids_ = snapshot.ids;
    host_vectors_ = snapshot.vectors;
    id_to_slot_.clear();
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        id_to_slot_[ids_[i]] = i;
    }
    count_ = ids_.size();
    sync_device_data_best_effort();
    return {};
}

} // namespace elips::gpu
