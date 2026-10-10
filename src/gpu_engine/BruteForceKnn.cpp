#include "elips/gpu_engine/BruteForceKnn.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>

#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Exact top-k of one distance row (length nb) by (value, index) ascending --
// the same deterministic tiebreak the Metal topk_select kernel uses, so the GPU
// and CPU paths agree label-for-label.
void cpu_select(const float* row, std::size_t nb, std::size_t k,
                std::int64_t* lbl_out, float* dist_out) {
    const std::size_t take = std::min(k, nb);
    std::vector<std::uint32_t> idx(nb);
    std::iota(idx.begin(), idx.end(), 0u);
    std::partial_sort(idx.begin(),
                      idx.begin() + static_cast<std::ptrdiff_t>(take), idx.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                          if (row[a] != row[b]) return row[a] < row[b];
                          return a < b;
                      });
    for (std::size_t j = 0; j < take; ++j) {
        lbl_out[j] = static_cast<std::int64_t>(idx[j]);
        dist_out[j] = row[idx[j]];
    }
    for (std::size_t j = take; j < k; ++j) {
        lbl_out[j] = -1;
        dist_out[j] = kInf;
    }
}

}  // namespace

std::expected<BfKnnResult, GpuError>
bf_knn(std::span<const float> queries, std::size_t nq,
       std::span<const float> database, std::size_t nb, std::size_t dim,
       std::size_t k, elips::Metric metric, GpuBackend* backend) {
    if (dim == 0 || k == 0 || nq == 0) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    if (queries.size() != nq * dim || database.size() != nb * dim) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }

    BfKnnResult out;
    out.nq = nq;
    out.k = k;
    out.labels.assign(nq * k, -1);
    out.distances.assign(nq * k, kInf);
    if (nb == 0) {
        return out;  // every slot is pad
    }

    // --- distance matrix (nq x nb) ---
    std::vector<float> dist(nq * nb);
    if (backend != nullptr) {
        auto r = backend->compute_distances_batch(queries, database, dist, nq, nb,
                                                  dim, metric);
        if (!r.has_value()) return std::unexpected(r.error());
    } else {
        for (std::size_t q = 0; q < nq; ++q) {
            for (std::size_t j = 0; j < nb; ++j) {
                dist[q * nb + j] =
                    elips::distance(metric, queries.subspan(q * dim, dim),
                                    database.subspan(j * dim, dim));
            }
        }
    }

    // --- top-k selection ---
    if (backend != nullptr) {
        std::vector<std::uint32_t> gi(nq * k);
        std::vector<float> gv(nq * k);
        auto r = backend->top_k(dist, gi, gv, nq, nb, k);
        if (!r.has_value()) return std::unexpected(r.error());
        const std::size_t take = std::min(k, nb);
        for (std::size_t q = 0; q < nq; ++q) {
            for (std::size_t j = 0; j < take; ++j) {
                out.labels[q * k + j] = static_cast<std::int64_t>(gi[q * k + j]);
                out.distances[q * k + j] = gv[q * k + j];
            }
            // slots [take, k) stay -1 / +inf from the initial fill
        }
    } else {
        for (std::size_t q = 0; q < nq; ++q) {
            cpu_select(dist.data() + q * nb, nb, k, out.labels.data() + q * k,
                       out.distances.data() + q * k);
        }
    }
    return out;
}

}  // namespace elips::gpu
