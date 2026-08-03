#ifndef ELIPS_GPU_ENGINE_GPU_INGESTION_PIPELINE_HPP
#define ELIPS_GPU_ENGINE_GPU_INGESTION_PIPELINE_HPP

#include <span>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/Expected.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/gpu_engine/GpuPort.hpp"

namespace elips::gpu {

class GpuIngestionPipeline {
public:
    explicit GpuIngestionPipeline(GpuPort& backend);

    [[nodiscard]] std::expected<void, GpuError>
    normalize_batch(std::span<float> vectors, size_t n, size_t dim);

    [[nodiscard]] std::expected<void, GpuError>
    quantize_batch(
        std::span<const float> vectors,
        std::span<uint8_t> codes_out,
        size_t n, size_t dim, size_t pq_dim, size_t pq_bits);

private:
    [[maybe_unused]] GpuPort& backend_;
};

} // namespace elips::gpu

#endif