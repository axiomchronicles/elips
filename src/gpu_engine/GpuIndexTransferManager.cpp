#include "elips/gpu_engine/GpuIndexTransferManager.hpp"

namespace elips::gpu {

std::expected<void, std::string> GpuIndexTransferManager::clone(
    const elips::IndexTransferPort& source,
    elips::IndexTransferPort& destination) {
    auto snapshot = source.export_snapshot();
    if (!snapshot.has_value()) {
        return std::unexpected(snapshot.error());
    }
    return destination.import_snapshot(*snapshot);
}

std::expected<void, GpuError> GpuIndexTransferManager::clone_cpu_to_gpu(
    const elips::IndexTransferPort& source,
    GpuIndexPort& destination) {
    auto result = clone(source, destination);
    if (!result.has_value()) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    return {};
}

std::expected<void, std::string> GpuIndexTransferManager::clone_gpu_to_cpu(
    const GpuIndexPort& source,
    elips::IndexTransferPort& destination) {
    return clone(source, destination);
}

}  // namespace elips::gpu
