#ifndef ELIPS_GPU_ENGINE_GPU_INDEX_PORT_HPP
#define ELIPS_GPU_ENGINE_GPU_INDEX_PORT_HPP

#include <expected>
#include <memory>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "elips/domain/RecordID.hpp"
#include "elips/domain/SearchResult.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuPort.hpp"
#include "elips/index_engine/IndexPort.hpp"
#include "elips/index_engine/IndexTransferPort.hpp"

namespace elips::gpu {

class GpuIndexPort : public elips::IndexPort, public elips::IndexTransferPort {
public:
    ~GpuIndexPort() override = default;

    [[nodiscard]] virtual std::expected<void, GpuError>
    build_from_batch(
        std::span<const float> vectors,
        std::span<const RecordID> ids,
        const GpuIndexBuildParams& params) = 0;

    [[nodiscard]] virtual std::expected<std::vector<std::vector<SearchResult>>, GpuError>
    search_batch(
        std::span<const float> queries,
        size_t k,
        size_t ef_search) const = 0;

    [[nodiscard]] virtual std::expected<void, GpuError>
    export_to_cpu_index(elips::IndexPort& cpu_index_out) const = 0;

    [[nodiscard]] virtual std::expected<void, GpuError>
    import_from_cpu_index(const elips::IndexPort& cpu_index) = 0;

    [[nodiscard]] virtual size_t device_bytes_used() const noexcept = 0;

    [[nodiscard]] virtual std::string_view backend_name() const noexcept = 0;
};

} // namespace elips::gpu

#endif
