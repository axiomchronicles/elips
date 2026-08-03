#include "elips/gpu_engine/GpuIngestionPipeline.hpp"
#include <cmath>

namespace elips::gpu {

GpuIngestionPipeline::GpuIngestionPipeline(GpuPort& backend)
    : backend_(backend) {}

std::expected<void, GpuError>
GpuIngestionPipeline::normalize_batch(std::span<float> vectors, size_t n, size_t dim) {
    for (size_t i = 0; i < n; ++i) {
        float* row = vectors.data() + i * dim;
        float mag = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            mag += row[j] * row[j];
        }
        mag = std::sqrt(mag);
        if (mag > 0.0f) {
            for (size_t j = 0; j < dim; ++j) {
                row[j] /= mag;
            }
        }
    }
    return {};
}

std::expected<void, GpuError>
GpuIngestionPipeline::quantize_batch(
    std::span<const float> vectors, std::span<uint8_t> codes_out,
    size_t n, size_t dim, size_t pq_dim, size_t pq_bits) {
    if (pq_dim == 0) pq_dim = dim / 2;
    const size_t sub_dim = dim / pq_dim;
    const size_t bytes_per_code = pq_bits / 8;

    for (size_t i = 0; i < n; ++i) {
        for (size_t m = 0; m < pq_dim; ++m) {
            const size_t offset = i * pq_dim * bytes_per_code + m * bytes_per_code;
            size_t code = 0;
            for (size_t j = 0; j < sub_dim; ++j) {
                const float v = vectors[i * dim + m * sub_dim + j];
                code = (code + static_cast<size_t>(std::abs(v) * 127.0f)) % 256;
            }
            codes_out[offset] = static_cast<uint8_t>(code & 0xFF);
        }
    }
    return {};
}

} // namespace elips::gpu