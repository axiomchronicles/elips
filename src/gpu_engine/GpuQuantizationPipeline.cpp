#include "elips/gpu_engine/GpuQuantizationPipeline.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <random>

namespace elips::gpu {

GpuQuantizationPipeline::GpuQuantizationPipeline(GpuPort& backend)
    : backend_(backend) {}

std::expected<void, GpuError>
GpuQuantizationPipeline::train_pq_codebook(
    std::span<const float> training_vectors, size_t n, size_t dim,
    size_t pq_dim, size_t n_lists, std::span<float> codebook_out) {
    if (n == 0 || pq_dim == 0 || dim == 0 || dim % pq_dim != 0 || n_lists == 0) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }

    size_t sub_dim = dim / pq_dim;
    size_t centroids_per_subspace = n_lists;
    size_t codebook_size = pq_dim * centroids_per_subspace * sub_dim;
    if (codebook_out.size() < codebook_size) {
        return std::unexpected(GpuError::InsufficientMemory);
    }

    for (size_t m = 0; m < pq_dim; ++m) {
        auto* codebook_base =
            codebook_out.data() + m * centroids_per_subspace * sub_dim;

        const size_t stride = std::max<std::size_t>(1, n / centroids_per_subspace);
        for (size_t c = 0; c < centroids_per_subspace; ++c) {
            const auto sample = std::min(n - 1, c * stride);
            const auto* example =
                training_vectors.data() + sample * dim + m * sub_dim;
            std::memcpy(codebook_base + c * sub_dim, example,
                        sub_dim * sizeof(float));
        }

        std::vector<size_t> assignments(n, 0);
        std::vector<float> sums(centroids_per_subspace * sub_dim, 0.0F);
        std::vector<size_t> counts(centroids_per_subspace, 0);

        for (size_t iter = 0; iter < 10; ++iter) {
            std::fill(assignments.begin(), assignments.end(), 0);
            std::fill(sums.begin(), sums.end(), 0.0F);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t row = 0; row < n; ++row) {
                const auto* vector =
                    training_vectors.data() + row * dim + m * sub_dim;
                float best_dist = std::numeric_limits<float>::max();
                size_t best_centroid = 0;
                for (size_t c = 0; c < centroids_per_subspace; ++c) {
                    const auto* centroid = codebook_base + c * sub_dim;
                    float distance = 0.0F;
                    for (size_t d = 0; d < sub_dim; ++d) {
                        const float delta = vector[d] - centroid[d];
                        distance += delta * delta;
                    }
                    if (distance < best_dist) {
                        best_dist = distance;
                        best_centroid = c;
                    }
                }
                assignments[row] = best_centroid;
                ++counts[best_centroid];
                auto* sum = sums.data() + best_centroid * sub_dim;
                for (size_t d = 0; d < sub_dim; ++d) {
                    sum[d] += vector[d];
                }
            }

            for (size_t c = 0; c < centroids_per_subspace; ++c) {
                auto* centroid = codebook_base + c * sub_dim;
                if (counts[c] == 0) {
                    const auto sample = std::min(n - 1, c % n);
                    const auto* fallback =
                        training_vectors.data() + sample * dim + m * sub_dim;
                    std::memcpy(centroid, fallback, sub_dim * sizeof(float));
                    continue;
                }
                for (size_t d = 0; d < sub_dim; ++d) {
                    centroid[d] = sums[c * sub_dim + d] /
                        static_cast<float>(counts[c]);
                }
            }
        }
    }
    return {};
}

std::expected<void, GpuError>
GpuQuantizationPipeline::encode_pq(
    std::span<const float> vectors, std::span<const float> codebook,
    size_t n, size_t dim, size_t pq_dim, std::span<uint8_t> codes_out) {
    size_t sub_dim = dim / pq_dim;
    size_t centroids_per_subspace = codebook.size() / (pq_dim * sub_dim);

    for (size_t i = 0; i < n; ++i) {
        for (size_t m = 0; m < pq_dim; ++m) {
            size_t best_c = 0;
            float best_dist = std::numeric_limits<float>::max();
            for (size_t c = 0; c < centroids_per_subspace; ++c) {
                float dist = 0;
                for (size_t d = 0; d < sub_dim; ++d) {
                    float diff = vectors[i * dim + m * sub_dim + d] -
                        codebook[m * centroids_per_subspace * sub_dim + c * sub_dim + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            codes_out[i * pq_dim + m] = static_cast<uint8_t>(best_c);
        }
    }
    return {};
}

} // namespace elips::gpu
