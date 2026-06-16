#ifndef ELIPS_INDEX_ENGINE_INDEX_SNAPSHOT_HPP
#define ELIPS_INDEX_ENGINE_INDEX_SNAPSHOT_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"

namespace elips {

enum class IndexSnapshotKind {
    unknown,
    exact,
    graph,
    gpu_brute_force,
    gpu_ivf_flat,
    gpu_ivf_pq,
    gpu_graph,
    gpu_hybrid,
    gpu_distributed,
};

struct IvfSnapshot {
    std::size_t n_lists{0};
    std::size_t n_probe{0};
    std::vector<float> centroids;
    std::vector<std::uint32_t> assignments;
};

struct PqSnapshot {
    std::uint32_t pq_dim{0};
    std::uint32_t pq_bits{0};
    std::vector<float> codebook;
    std::vector<std::uint8_t> codes;
};

// Canonical transfer format between CPU and GPU indexes. Raw vectors remain the
// source of truth so heterogeneous indexes can rebuild when their native
// topology cannot be copied verbatim.
struct IndexSnapshot {
    IndexSnapshotKind kind{IndexSnapshotKind::unknown};
    Metric metric{Metric::cosine};
    std::uint16_t dimension{0};
    std::vector<RecordID> ids;
    std::vector<float> vectors;
    std::optional<IvfSnapshot> ivf;
    std::optional<PqSnapshot> pq;
};

}  // namespace elips

#endif  // ELIPS_INDEX_ENGINE_INDEX_SNAPSHOT_HPP
