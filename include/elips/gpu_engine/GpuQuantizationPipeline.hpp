#ifndef ELIPS_GPU_ENGINE_GPU_QUANTIZATION_PIPELINE_HPP
#define ELIPS_GPU_ENGINE_GPU_QUANTIZATION_PIPELINE_HPP

#include <span>
#include <vector>

#include "elips/domain/Expected.hpp"
#include "elips/gpu_engine/GpuPort.hpp"

namespace elips::gpu {

class GpuQuantizationPipeline {
public:
    explicit GpuQuantizationPipeline(GpuPort& backend);

    [[nodiscard]] std::expected<void, GpuError>
    train_pq_codebook(
        std::span<const float> training_vectors,
        size_t n, size_t dim,
        size_t pq_dim, size_t n_lists,
        std::span<float> codebook_out);

    [[nodiscard]] std::expected<void, GpuError>
    encode_pq(
        std::span<const float> vectors,
        std::span<const float> codebook,
        size_t n, size_t dim, size_t pq_dim,
        std::span<uint8_t> codes_out);

private:
    GpuPort& backend_;
};

} // namespace elips::gpu

#endif