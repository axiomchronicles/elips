// Phase 7.2 real-Metal parity: the GPU IVFFlat search path (coarse probe +
// two-pass GPU list-scan/merge kernels, elips_ivf_scan_list / elips_ivf_merge_lists)
// checked against a CPU IndexIVFFlat oracle built on the SAME clustered data.
//
// At nprobe == nlist both indexes scan every list, so the GPU result must match
// the CPU exhaustive top-k exactly (labels equal except at genuine fp near-ties;
// distances within tolerance). At nprobe < nlist the two quantizers probe their
// own nearest lists, so we assert recall of the GPU top-k against the CPU top-k.
// Runs on the REAL Metal backend (never a mock); skips only without a device.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <unordered_set>
#include <vector>

#include "elips/gpu_engine/BackendSelector.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/gpu_engine/GpuIVFFlatIndex.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/IndexIVFFlat.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu {
namespace {

constexpr float kTol = 6e-3f;  // fp32 tolerance (per _faiss_3I)

bool near(float a, float b) {
    return std::fabs(a - b) <= kTol * (1.0f + std::fabs(b));
}

void l2_normalize(std::span<float> row) {
    float n = 0.0f;
    for (float v : row) n += v * v;
    n = std::sqrt(n);
    if (n > 0.0f)
        for (float& v : row) v /= n;
}

// n points over `nclusters` well-separated Gaussian blobs stored as CONTIGUOUS
// equal blocks (block l == cluster l). With n a multiple of nclusters this
// aligns with the GPU trainer's strided centroid seeding (one seed per block),
// so the CPU and GPU quantizers recover the SAME partition -- a prerequisite
// for probing identical lists at matched nprobe. Normalized for cosine.
std::vector<float> make_clustered(size_t n, size_t dim, size_t nclusters,
                                  uint64_t seed, bool normalize) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> center(-10.0f, 10.0f);
    std::normal_distribution<float> noise(0.0f, 0.15f);

    std::vector<float> centers(nclusters * dim);
    for (auto& c : centers) c = center(rng);

    const size_t block = n / nclusters;
    std::vector<float> v(n * dim);
    for (size_t i = 0; i < n; ++i) {
        const size_t c = std::min(nclusters - 1, i / block);  // contiguous blocks
        for (size_t d = 0; d < dim; ++d)
            v[i * dim + d] = centers[c * dim + d] + noise(rng);
        if (normalize) l2_normalize(std::span<float>{v.data() + i * dim, dim});
    }
    return v;
}

std::unique_ptr<GpuBackend> make_real_metal_backend() {
    GpuDeviceManager manager;
    BackendSelector selector;
    GpuConfig config;
    config.policy = GpuPolicy::Auto;
    auto sel = selector.select(config, manager.probe_all_devices());
    if (!sel.has_value()) return nullptr;
    auto backend = std::move(*sel);
    if (backend->device_info().backend != "metal") return nullptr;
    return backend;
}

struct Case {
    Metric metric;
    size_t dim;
    size_t nlist;
    size_t nprobe;
    size_t k;
};

double run_case(GpuBackend& be, const Case& tc, uint64_t seed) {
    const bool normalize = (tc.metric == Metric::cosine);
    const size_t n = 5120;  // multiple of nlist (16) -> aligned trainer seeding
    const size_t nq = 32;

    auto db = make_clustered(n, tc.dim, tc.nlist, seed, normalize);
    // Same seed -> identical cluster centers, so queries sit near the SAME
    // blobs the database was drawn from (but are not database points).
    auto q = make_clustered(nq, tc.dim, tc.nlist, seed, normalize);

    std::vector<RecordID> ids;
    ids.reserve(n);
    for (size_t i = 0; i < n; ++i) ids.push_back(RecordID::generate());

    // CPU oracle: IndexIVFFlat on identical data.
    IndexIVFFlat cpu(tc.metric, static_cast<uint16_t>(tc.dim), tc.nlist);
    cpu.train(db, n);
    for (size_t i = 0; i < n; ++i)
        cpu.add_with_ids(ids[i], std::span<const float>{db.data() + i * tc.dim, tc.dim});
    cpu.nprobe = tc.nprobe;

    // GPU IVF path on the real Metal backend.
    GpuConfig gconfig;
    gconfig.algorithm = GpuIndexAlgorithm::IvfFlat;
    gconfig.ivf_pq_params.n_lists = tc.nlist;
    gconfig.default_ef_search_gpu = tc.nprobe;
    GpuIVFFlatIndex gpu(be, tc.metric, static_cast<uint16_t>(tc.dim), gconfig);
    auto built = gpu.build_from_batch(db, ids, GpuIndexBuildParams{});
    EXPECT_TRUE(built.has_value());

    const bool exhaustive = (tc.nprobe >= tc.nlist);
    double recall_sum = 0.0;

    for (size_t i = 0; i < nq; ++i) {
        const std::span<const float> query{q.data() + i * tc.dim, tc.dim};
        const auto cpu_hits = cpu.search(query, tc.k);
        const auto gpu_hits = gpu.search(query, tc.k);

        if (exhaustive) {
            // Both scan every list -> exact top-k parity.
            EXPECT_EQ(gpu_hits.size(), cpu_hits.size())
                << "metric=" << static_cast<int>(tc.metric) << " dim=" << tc.dim;
            const size_t m = std::min(gpu_hits.size(), cpu_hits.size());
            for (size_t j = 0; j < m; ++j) {
                if (gpu_hits[j].first != cpu_hits[j].first) {
                    EXPECT_TRUE(near(gpu_hits[j].second, cpu_hits[j].second))
                        << "non-tie label divergence metric="
                        << static_cast<int>(tc.metric) << " dim=" << tc.dim
                        << " q=" << i << " pos=" << j;
                }
                EXPECT_TRUE(near(gpu_hits[j].second, cpu_hits[j].second))
                    << "distance mismatch metric=" << static_cast<int>(tc.metric)
                    << " dim=" << tc.dim << " q=" << i << " pos=" << j;
            }
        }

        std::unordered_set<RecordID> cpu_set;
        for (const auto& [id, d] : cpu_hits) cpu_set.insert(id);
        size_t match = 0;
        for (const auto& [id, d] : gpu_hits)
            if (cpu_set.count(id)) ++match;
        recall_sum += cpu_hits.empty() ? 1.0
                                       : static_cast<double>(match) / cpu_hits.size();
    }
    return recall_sum / nq;
}

TEST(MetalIvfParity, gpu_ivf_matches_cpu_ivfflat) {
    auto backend = make_real_metal_backend();
    if (!backend) GTEST_SKIP() << "No Metal device available";
    ASSERT_EQ(backend->device_info().backend, "metal");

    const Metric metrics[] = {Metric::euclidean, Metric::dot_product, Metric::cosine};
    const size_t dims[] = {32, 128};
    const size_t nlist = 16;
    const size_t nprobes[] = {1, 8, nlist};

    uint64_t seed = 7020202u;
    for (Metric m : metrics) {
        for (size_t dim : dims) {
            for (size_t np : nprobes) {
                Case tc{m, dim, nlist, np, /*k=*/10};
                const double recall = run_case(*backend, tc, seed++);
                // Exhaustive probing (nprobe==nlist) is exact up to fp near-ties
                // at the k-boundary, which the per-hit near() check above already
                // tolerates; the residual recall dip is only those boundary swaps.
                // Below nlist, recall reflects shared-partition IVF agreement.
                const double floor = (np >= nlist) ? 0.99 : (np >= 8 ? 0.85 : 0.60);
                EXPECT_GE(recall, floor)
                    << "metric=" << static_cast<int>(m) << " dim=" << dim
                    << " nprobe=" << np << " recall=" << recall;
                RecordProperty("recall", std::to_string(recall));
                std::cout << "[ivf-parity] metric=" << static_cast<int>(m)
                          << " dim=" << dim << " nprobe=" << np
                          << " recall=" << recall << "\n";
            }
        }
    }
}

}  // namespace
}  // namespace elips::gpu
