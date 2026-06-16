#ifndef ELIPS_GPU_ENGINE_DETAIL_IVF_INDEX_STATE_HPP
#define ELIPS_GPU_ENGINE_DETAIL_IVF_INDEX_STATE_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/gpu_engine/GpuBuffer.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuPort.hpp"
#include "elips/index_engine/IndexSnapshot.hpp"

namespace elips::gpu::detail {

class IvfIndexState {
public:
    struct CandidateSet {
        std::vector<RecordID> ids;
        std::vector<float> vectors;
        std::vector<std::uint32_t> source_slots;
        std::vector<std::uint32_t> source_lists;
    };

    IvfIndexState(elips::Metric metric, std::uint16_t dimension,
                  const GpuConfig& config);
    ~IvfIndexState();

    IvfIndexState(const IvfIndexState&) = delete;
    IvfIndexState& operator=(const IvfIndexState&) = delete;
    IvfIndexState(IvfIndexState&&) = delete;
    IvfIndexState& operator=(IvfIndexState&&) = delete;

    void insert(GpuPort& backend, const RecordID& id, std::span<const float> vector);
    void remove(GpuPort& backend, const RecordID& id);

    [[nodiscard]] std::expected<void, std::string>
    build_from_batch(GpuPort& backend, std::span<const float> vectors,
                     std::span<const RecordID> ids);

    [[nodiscard]] std::expected<void, std::string>
    import_snapshot(GpuPort& backend, const elips::IndexSnapshot& snapshot);

    [[nodiscard]] elips::IndexSnapshot
    export_snapshot(elips::IndexSnapshotKind kind) const;

    [[nodiscard]] std::expected<std::vector<std::uint32_t>, GpuError>
    probe_lists(GpuPort& backend, std::span<const float> query,
                std::size_t nprobe) const;

    [[nodiscard]] std::expected<std::vector<std::vector<std::uint32_t>>, GpuError>
    probe_lists_batch(GpuPort& backend, std::span<const float> queries,
                      std::size_t nprobe) const;

    [[nodiscard]] CandidateSet gather_candidates(
        const std::vector<std::uint32_t>& lists) const;

    [[nodiscard]] CandidateSet all_candidates() const;

    [[nodiscard]] std::size_t size() const noexcept { return ids_.size(); }
    [[nodiscard]] bool trained() const noexcept { return trained_; }
    [[nodiscard]] std::size_t active_list_count() const noexcept {
        return active_n_lists_;
    }
    [[nodiscard]] std::size_t nprobe() const noexcept { return nprobe_; }
    [[nodiscard]] std::size_t device_bytes_used() const noexcept;

    [[nodiscard]] const std::vector<RecordID>& ids() const noexcept { return ids_; }
    [[nodiscard]] const std::vector<float>& host_vectors() const noexcept {
        return host_vectors_;
    }
    [[nodiscard]] const std::vector<float>& centroids() const noexcept {
        return centroids_;
    }
    [[nodiscard]] const std::vector<std::uint32_t>& assignments() const noexcept {
        return vector_to_list_;
    }

private:
    void clear() noexcept;
    void validate_dimension(std::span<const float> vector,
                            std::string_view label) const;
    void rebuild_id_map();
    void rebuild_list_members();
    void maybe_retrain_after_growth(GpuPort& backend);
    void rebuild_from_current_data(GpuPort& backend, std::size_t n_lists);
    void sync_device_buffers_best_effort(GpuPort& backend) noexcept;

    [[nodiscard]] std::vector<float> train_centroids(
        std::span<const float> vectors,
        std::size_t nvecs,
        std::size_t n_lists) const;

    [[nodiscard]] std::vector<std::uint32_t> assign_vectors_to_centroids(
        std::span<const float> vectors,
        std::size_t nvecs,
        std::span<const float> centroids,
        std::size_t n_lists) const;

    elips::Metric metric_;
    std::uint16_t dimension_;
    GpuConfig config_;
    std::size_t requested_n_lists_;
    std::size_t active_n_lists_{0};
    std::size_t nprobe_;
    bool trained_{false};

    std::vector<RecordID> ids_;
    std::vector<float> host_vectors_;
    std::vector<float> centroids_;
    std::vector<std::uint32_t> vector_to_list_;
    std::vector<std::vector<std::uint32_t>> list_members_;
    std::unordered_map<RecordID, std::size_t> id_to_slot_;

    GpuBuffer vector_buffer_;
    GpuBuffer centroid_buffer_;
};

}  // namespace elips::gpu::detail

#endif  // ELIPS_GPU_ENGINE_DETAIL_IVF_INDEX_STATE_HPP
