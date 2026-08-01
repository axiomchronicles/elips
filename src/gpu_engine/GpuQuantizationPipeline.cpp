#include "elips/gpu_engine/GpuQuantizationPipeline.hpp"

#include <algorithm>

#include "elips/quant_engine/detail/KMeans.hpp"

namespace elips::gpu {

GpuQuantizationPipeline::GpuQuantizationPipeline(GpuPort& backend)
    : backend_(backend) {}

// Both entry points delegate to elips::quant::detail, which owns the only
// k-means implementation in the codebase. This pipeline predates that module and
// carried its own copy; the duplicate has been removed rather than left to drift
// against the CPU codecs that now share a code layout with it.
std::expected<void, GpuError>
GpuQuantizationPipeline::train_pq_codebook(
    std::span<const float> training_vectors, size_t n, size_t dim,
    size_t pq_dim, size_t n_lists, std::span<float> codebook_out) {
    if (n == 0 || pq_dim == 0 || dim == 0 || dim % pq_dim != 0 || n_lists == 0) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }

    const size_t sub_dim = dim / pq_dim;
    const size_t codebook_size = pq_dim * n_lists * sub_dim;
    if (codebook_out.size() < codebook_size) {
        return std::unexpected(GpuError::InsufficientMemory);
    }

    quant::detail::train_pq_codebook(training_vectors, n, dim, pq_dim, n_lists,
                                     /*iters=*/10, codebook_out);
    return {};
}

std::expected<void, GpuError>
GpuQuantizationPipeline::encode_pq(
    std::span<const float> vectors, std::span<const float> codebook, size_t n,
    size_t dim, size_t pq_dim, std::span<uint8_t> codes_out) {
    if (pq_dim == 0 || dim == 0 || dim % pq_dim != 0) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }

    const size_t sub_dim = dim / pq_dim;
    const size_t ksub = codebook.size() / (pq_dim * sub_dim);
    if (ksub == 0 || ksub > 256) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    if (codes_out.size() < n * pq_dim) {
        return std::unexpected(GpuError::InsufficientMemory);
    }

    quant::detail::encode_pq(vectors, codebook, n, dim, pq_dim, ksub, codes_out);
    return {};
}

}  // namespace elips::gpu
