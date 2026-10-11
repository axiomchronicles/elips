#ifndef ELIPS_GPU_ENGINE_BRUTE_FORCE_KNN_HPP
#define ELIPS_GPU_ENGINE_BRUTE_FORCE_KNN_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/Expected.hpp"
#include "elips/gpu_engine/GpuBackend.hpp"

namespace elips::gpu {

// Flat row-major (nq*k) top-k result; id == database row index. Slots past nb
// (when k > nb) hold label -1 and a +inf sentinel distance, matching the FAISS
// heap-padding contract. Distances are ordering-normalized (smaller = closer).
struct BfKnnResult {
    std::vector<std::int64_t> labels;     // nq*k, row-major, -1 pad
    std::vector<float>        distances;  // nq*k, row-major, +inf pad
    std::size_t nq{0};
    std::size_t k{0};
};

// Brute-force exact k-NN over raw row-major arrays -- no index build.
//   queries:  nq * dim floats, database: nb * dim floats.
// When `backend` is non-null the backend's real GPU distance + top-k kernels do
// the work; when null an exact CPU path runs. Both select the k smallest
// distances with a (distance, row-index) ascending tiebreak, so results match.
// Cosine follows the ELIPS convention: inputs are assumed L2-normalized.
[[nodiscard]] std::expected<BfKnnResult, GpuError>
bf_knn(std::span<const float> queries, std::size_t nq,
       std::span<const float> database, std::size_t nb,
       std::size_t dim, std::size_t k, elips::Metric metric,
       GpuBackend* backend);

}  // namespace elips::gpu

#endif  // ELIPS_GPU_ENGINE_BRUTE_FORCE_KNN_HPP
