#ifndef ELIPS_GPU_ENGINE_GPU_IVF_FLAT_INDEX_HPP
#define ELIPS_GPU_ENGINE_GPU_IVF_FLAT_INDEX_HPP

#include <memory>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/Expected.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/gpu_engine/GpuIndex.hpp"
#include "elips/gpu_engine/GpuBackend.hpp"

namespace elips::gpu {
namespace detail {
class IvfIndexState;
}

class GpuIVFFlatIndex final : public GpuIndex {
public:
    GpuIVFFlatIndex(GpuBackend& backend, elips::Metric metric, uint16_t dimension,
                    const GpuConfig& config);
    ~GpuIVFFlatIndex() override;

    void add_with_ids(const RecordID& id, std::span<const float> vector) override;
    void remove(const RecordID& id) override;
    [[nodiscard]] std::vector<Hit> search(std::span<const float> query,
                                          std::size_t k) const override;
    [[nodiscard]] std::size_t ntotal() const noexcept override;
    [[nodiscard]] std::string_view type_name() const noexcept override;

    [[nodiscard]] std::expected<void, GpuError>
    build_from_batch(std::span<const float> vectors, std::span<const RecordID> ids,
                     const GpuIndexBuildParams& params) override;

    [[nodiscard]] std::expected<std::vector<std::vector<SearchResult>>, GpuError>
    search_batch(std::span<const float> queries, size_t k, size_t ef_search) const override;

    [[nodiscard]] std::expected<void, GpuError>
    export_to_cpu_index(elips::Index& cpu_index_out) const override;

    [[nodiscard]] std::expected<void, GpuError>
    import_from_cpu_index(const elips::Index& cpu_index) override;

    [[nodiscard]] size_t device_bytes_used() const noexcept override;
    [[nodiscard]] std::string_view backend_name() const noexcept override;

    [[nodiscard]] std::expected<elips::IndexSnapshot, std::string>
    export_snapshot() const override;

    [[nodiscard]] std::expected<void, std::string>
    import_snapshot(const elips::IndexSnapshot& snapshot) override;

private:
    [[nodiscard]] std::vector<Hit> rank_candidates(
        std::span<const float> query, std::size_t k,
        const std::vector<RecordID>& candidate_ids,
        const std::vector<float>& candidate_vectors) const;

    // Two-pass GPU IVF scan+merge path. Uses source_lists to recover per-list
    // boundaries and drives the backend ivf_scan kernel; falls back to
    // rank_candidates() when the backend has no IVF-scan capability.
    [[nodiscard]] std::vector<Hit> rank_candidates_ivf(
        std::span<const float> query, std::size_t k,
        const std::vector<RecordID>& candidate_ids,
        const std::vector<float>& candidate_vectors,
        const std::vector<std::uint32_t>& source_lists) const;

    GpuBackend& backend_;
    elips::Metric metric_;
    uint16_t dimension_;
    GpuConfig config_;
    std::string backend_name_;
    std::unique_ptr<detail::IvfIndexState> state_;
};

} // namespace elips::gpu

#endif
