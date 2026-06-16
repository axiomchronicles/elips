#ifndef ELIPS_GPU_ENGINE_GPU_GRAPH_INDEX_HPP
#define ELIPS_GPU_ENGINE_GPU_GRAPH_INDEX_HPP

#include <expected>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/gpu_engine/GpuIndexPort.hpp"
#include "elips/gpu_engine/GpuPort.hpp"

namespace elips {
class HierarchicalGraphIndex;
}

namespace elips::gpu {

class GpuGraphIndex final : public GpuIndexPort {
public:
    GpuGraphIndex(GpuPort& backend, elips::Metric metric, uint16_t dimension,
                  const GpuConfig& config);
    ~GpuGraphIndex() override;

    void insert(const RecordID& id, std::span<const float> vector) override;
    void remove(const RecordID& id) override;
    [[nodiscard]] std::vector<Hit> search(std::span<const float> query,
                                          std::size_t k) const override;
    [[nodiscard]] std::size_t size() const noexcept override;
    [[nodiscard]] std::string_view type_name() const noexcept override;

    [[nodiscard]] std::expected<void, GpuError>
    build_from_batch(std::span<const float> vectors, std::span<const RecordID> ids,
                     const GpuIndexBuildParams& params) override;

    [[nodiscard]] std::expected<std::vector<std::vector<SearchResult>>, GpuError>
    search_batch(std::span<const float> queries, size_t k, size_t ef_search) const override;

    [[nodiscard]] std::expected<void, GpuError>
    export_to_cpu_index(elips::IndexPort& cpu_index_out) const override;

    [[nodiscard]] std::expected<void, GpuError>
    import_from_cpu_index(const elips::IndexPort& cpu_index) override;

    [[nodiscard]] size_t device_bytes_used() const noexcept override;
    [[nodiscard]] std::string_view backend_name() const noexcept override;

    [[nodiscard]] std::expected<elips::IndexSnapshot, std::string>
    export_snapshot() const override;

    [[nodiscard]] std::expected<void, std::string>
    import_snapshot(const elips::IndexSnapshot& snapshot) override;

private:
    void release_graph_data() noexcept;
    void sync_device_data_best_effort() noexcept;

    GpuPort& backend_;
    elips::Metric metric_;
    uint16_t dimension_;
    size_t count_{0};
    GpuBuffer graph_data_;
    std::vector<RecordID> ids_;
    std::vector<float> host_vectors_;
    std::unordered_map<RecordID, std::size_t> id_to_slot_;
    std::unique_ptr<elips::HierarchicalGraphIndex> cpu_graph_;
    std::string backend_name_;
    GpuConfig config_;
};

} // namespace elips::gpu

#endif
