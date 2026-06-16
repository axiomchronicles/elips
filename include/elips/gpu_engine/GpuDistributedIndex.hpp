#ifndef ELIPS_GPU_ENGINE_GPU_DISTRIBUTED_INDEX_HPP
#define ELIPS_GPU_ENGINE_GPU_DISTRIBUTED_INDEX_HPP

#include <atomic>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "elips/gpu_engine/GpuIndexPort.hpp"

namespace elips::gpu {

enum class DistributedMode {
    replicate,
    shard,
};

class GpuDistributedIndex final : public GpuIndexPort {
public:
    GpuDistributedIndex(std::vector<std::unique_ptr<GpuIndexPort>> indexes,
                        DistributedMode mode);
    ~GpuDistributedIndex() override = default;

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
    [[nodiscard]] std::vector<Hit> merge_hits(
        const std::vector<std::vector<Hit>>& shard_hits, std::size_t k) const;

    std::vector<std::unique_ptr<GpuIndexPort>> indexes_;
    DistributedMode mode_;
    mutable std::atomic<std::size_t> next_replica_{0};
    std::string backend_name_{"distributed"};
};

}  // namespace elips::gpu

#endif  // ELIPS_GPU_ENGINE_GPU_DISTRIBUTED_INDEX_HPP
