#ifndef ELIPS_GPU_ENGINE_GPU_SEARCH_PIPELINE_HPP
#define ELIPS_GPU_ENGINE_GPU_SEARCH_PIPELINE_HPP

#include <span>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/Expected.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/domain/SearchResult.hpp"
#include "elips/gpu_engine/GpuPort.hpp"

namespace elips::gpu {

class GpuSearchPipeline {
public:
    explicit GpuSearchPipeline(GpuPort& backend);

    [[nodiscard]] std::expected<std::vector<std::vector<elips::SearchResult>>, GpuError>
    batch_search(
        std::span<const float> queries,
        size_t nq,
        std::span<const float> database,
        std::span<const RecordID> db_ids,
        size_t nb,
        size_t dim,
        size_t k,
        elips::Metric metric);

private:
    GpuPort& backend_;
};

} // namespace elips::gpu

#endif