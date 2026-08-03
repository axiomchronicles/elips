#include "elips/gpu_engine/GpuSearchPipeline.hpp"
#include <cstring>

namespace elips::gpu {

GpuSearchPipeline::GpuSearchPipeline(GpuPort& backend)
    : backend_(backend) {}

std::expected<std::vector<std::vector<elips::SearchResult>>, GpuError>
GpuSearchPipeline::batch_search(
    std::span<const float> queries, size_t nq,
    std::span<const float> database, std::span<const RecordID> db_ids,
    size_t nb, size_t dim, size_t k, elips::Metric metric) {
    std::vector<float> distances(nq * nb);
    auto dist_result = backend_.compute_distances_batch(queries, database, distances, nq, nb, dim, metric);
    if (!dist_result.has_value()) return std::unexpected(dist_result.error());

    std::vector<uint32_t> indices(nq * k);
    std::vector<float> values(nq * k);
    auto topk_result = backend_.top_k(distances, indices, values, nq, nb, k);
    if (!topk_result.has_value()) return std::unexpected(topk_result.error());

    std::vector<std::vector<elips::SearchResult>> results(nq);
    for (size_t q = 0; q < nq; ++q) {
        results[q].reserve(k);
        for (size_t j = 0; j < k; ++j) {
            const uint32_t idx = indices[q * k + j];
            if (idx >= nb) break;
            results[q].push_back(elips::SearchResult{
                db_ids[idx],
                values[q * k + j],
                {},
                std::nullopt,
                std::nullopt,
                std::nullopt,
                quant::CodecId::none
            });
        }
    }
    return results;
}

} // namespace elips::gpu