#include "elips/gpu_engine/GpuHybridIndex.hpp"

#include <memory>

#include "elips/gpu_engine/GpuBruteForceIndex.hpp"
#include "elips/gpu_engine/GpuGraphIndex.hpp"
#include "elips/gpu_engine/GpuIVFFlatIndex.hpp"
#include "elips/gpu_engine/GpuIVFPQIndex.hpp"
#include "elips/gpu_engine/GpuIndexTransferManager.hpp"

namespace elips::gpu {
namespace {

std::unique_ptr<GpuIndexPort> make_gpu_index(GpuPort& backend, elips::Metric metric,
                                             uint16_t dimension,
                                             const GpuConfig& config) {
    switch (config.algorithm) {
        case GpuIndexAlgorithm::BruteForce:
            return std::make_unique<GpuBruteForceIndex>(
                backend, metric, dimension, config);
        case GpuIndexAlgorithm::IvfFlat:
            return std::make_unique<GpuIVFFlatIndex>(
                backend, metric, dimension, config);
        case GpuIndexAlgorithm::IvfPq:
            return std::make_unique<GpuIVFPQIndex>(
                backend, metric, dimension, config);
        case GpuIndexAlgorithm::CagraGraph:
            return std::make_unique<GpuGraphIndex>(
                backend, metric, dimension, config);
        case GpuIndexAlgorithm::Auto:
            if (backend.device_info().supports_ivf_pq) {
                return std::make_unique<GpuIVFFlatIndex>(
                    backend, metric, dimension, config);
            }
            return std::make_unique<GpuBruteForceIndex>(
                backend, metric, dimension, config);
    }
    return std::make_unique<GpuBruteForceIndex>(backend, metric, dimension, config);
}

}  // namespace

GpuHybridIndex::GpuHybridIndex(GpuPort& backend,
                               std::unique_ptr<elips::IndexPort> cpu_index,
                               elips::Metric metric,
                               uint16_t dimension,
                               const GpuConfig& config)
    : cpu_index_(std::move(cpu_index))
    , gpu_index_(make_gpu_index(backend, metric, dimension, config))
{}

void GpuHybridIndex::insert(const RecordID& id, std::span<const float> vector) {
    cpu_index_->insert(id, vector);
    gpu_index_->insert(id, vector);
}

void GpuHybridIndex::remove(const RecordID& id) {
    cpu_index_->remove(id);
    gpu_index_->remove(id);
}

std::vector<IndexPort::Hit> GpuHybridIndex::search(std::span<const float> query,
                                                    std::size_t k) const {
    const auto gpu_hits = gpu_index_->search(query, k);
    if (!gpu_hits.empty() || gpu_index_->size() == 0 || k == 0) {
        return gpu_hits;
    }
    return cpu_index_->search(query, k);
}

std::size_t GpuHybridIndex::size() const noexcept { return cpu_index_->size(); }
std::string_view GpuHybridIndex::type_name() const noexcept { return "gpu_hybrid"; }

std::expected<void, GpuError>
GpuHybridIndex::build_from_batch(std::span<const float> vectors,
                                 std::span<const RecordID> ids,
                                 const GpuIndexBuildParams& params) {
    auto build = gpu_index_->build_from_batch(vectors, ids, params);
    if (!build.has_value()) {
        return build;
    }

    auto* cpu_transfer = dynamic_cast<elips::IndexTransferPort*>(cpu_index_.get());
    if (!cpu_transfer) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    auto sync = GpuIndexTransferManager::clone_gpu_to_cpu(*gpu_index_, *cpu_transfer);
    if (!sync.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return {};
}

std::expected<std::vector<std::vector<SearchResult>>, GpuError>
GpuHybridIndex::search_batch(std::span<const float> queries, size_t k, size_t ef_search) const {
    return gpu_index_->search_batch(queries, k, ef_search);
}

std::expected<void, GpuError>
GpuHybridIndex::export_to_cpu_index(elips::IndexPort& cpu_index_out) const {
    auto* transfer = dynamic_cast<elips::IndexTransferPort*>(&cpu_index_out);
    if (!transfer) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    auto* cpu_source = dynamic_cast<elips::IndexTransferPort*>(cpu_index_.get());
    if (!cpu_source) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    auto cloned = GpuIndexTransferManager::clone(*cpu_source, *transfer);
    if (!cloned.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return {};
}

std::expected<void, GpuError>
GpuHybridIndex::import_from_cpu_index(const elips::IndexPort& cpu_index) {
    const auto* cpu_transfer =
        dynamic_cast<const elips::IndexTransferPort*>(&cpu_index);
    auto* cpu_mirror_transfer =
        dynamic_cast<elips::IndexTransferPort*>(cpu_index_.get());
    if (!cpu_transfer || !cpu_mirror_transfer) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }

    auto mirror = GpuIndexTransferManager::clone(*cpu_transfer, *cpu_mirror_transfer);
    if (!mirror.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return GpuIndexTransferManager::clone_cpu_to_gpu(*cpu_transfer, *gpu_index_);
}

size_t GpuHybridIndex::device_bytes_used() const noexcept {
    return gpu_index_->device_bytes_used();
}

std::string_view GpuHybridIndex::backend_name() const noexcept {
    return gpu_index_->backend_name();
}

std::expected<elips::IndexSnapshot, std::string>
GpuHybridIndex::export_snapshot() const {
    auto* transfer = dynamic_cast<const elips::IndexTransferPort*>(cpu_index_.get());
    if (!transfer) {
        return std::unexpected("cpu mirror does not support snapshot export");
    }

    auto snapshot = transfer->export_snapshot();
    if (!snapshot.has_value()) {
        return snapshot;
    }
    snapshot->kind = elips::IndexSnapshotKind::gpu_hybrid;
    return snapshot;
}

std::expected<void, std::string>
GpuHybridIndex::import_snapshot(const elips::IndexSnapshot& snapshot) {
    auto* cpu_transfer = dynamic_cast<elips::IndexTransferPort*>(cpu_index_.get());
    if (!cpu_transfer) {
        return std::unexpected("cpu mirror does not support snapshot import");
    }
    auto cpu_result = cpu_transfer->import_snapshot(snapshot);
    if (!cpu_result.has_value()) {
        return cpu_result;
    }

    auto gpu_result = gpu_index_->import_snapshot(snapshot);
    if (!gpu_result.has_value()) {
        return gpu_result;
    }
    return {};
}

} // namespace elips::gpu
