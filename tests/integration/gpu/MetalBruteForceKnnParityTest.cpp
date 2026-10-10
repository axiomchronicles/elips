// Real-Metal parity tests for Phase 7.3: the standalone bf_knn free function
// and the GpuBruteForceIndex search path, both driven by the ACTUAL Metal
// backend on this machine (never RecordingBackend), checked against exact CPU
// oracles. Skips only when no Metal device is present; on Apple Silicon it runs.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include "elips/gpu_engine/BackendSelector.hpp"
#include "elips/gpu_engine/BruteForceKnn.hpp"
#include "elips/gpu_engine/GpuBruteForceIndex.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/IndexFlat.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu {
namespace {

constexpr float kRelTol = 6e-3f;  // fp32 relative tolerance (per _faiss_3I/3H)

bool near(float a, float b) {
    return std::fabs(a - b) <= kRelTol * (1.0f + std::fabs(b));
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

std::vector<float> make_data(size_t n, size_t dim, uint64_t seed, bool normalize) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = dist(rng);
    if (normalize) {
        for (size_t i = 0; i < n; ++i) {
            float norm = 0.0f;
            for (size_t d = 0; d < dim; ++d) norm += v[i * dim + d] * v[i * dim + d];
            norm = std::sqrt(norm);
            if (norm > 0.0f)
                for (size_t d = 0; d < dim; ++d) v[i * dim + d] /= norm;
        }
    }
    return v;
}

// bf_knn on the real Metal backend must agree with the CPU bf_knn path: exact
// labels except at genuine fp near-ties, matching distances within tolerance,
// and -1 padding for k > nb.
void run_bf_case(GpuBackend& be, Metric metric, size_t nq, size_t nb, size_t dim,
                 size_t k, uint64_t seed) {
    const bool normalize = (metric == Metric::cosine);
    auto db = make_data(nb, dim, seed, normalize);
    auto q = make_data(nq, dim, seed ^ 0x9E3779B97F4A7C15ULL, normalize);

    auto gpu = bf_knn(std::span<const float>{q}, nq, std::span<const float>{db},
                      nb, dim, k, metric, &be);
    auto cpu = bf_knn(std::span<const float>{q}, nq, std::span<const float>{db},
                      nb, dim, k, metric, nullptr);
    ASSERT_TRUE(gpu.has_value());
    ASSERT_TRUE(cpu.has_value());

    const size_t take = std::min(k, nb);
    for (size_t i = 0; i < nq; ++i) {
        for (size_t j = 0; j < take; ++j) {
            const size_t s = i * k + j;
            if (gpu->labels[s] != cpu->labels[s]) {
                EXPECT_TRUE(near(gpu->distances[s], cpu->distances[s]))
                    << "non-tie label divergence metric=" << static_cast<int>(metric)
                    << " dim=" << dim << " nb=" << nb << " k=" << k << " q=" << i
                    << " pos=" << j;
            }
            EXPECT_TRUE(near(gpu->distances[s], cpu->distances[s]))
                << "distance mismatch metric=" << static_cast<int>(metric)
                << " dim=" << dim << " nb=" << nb << " k=" << k << " pos=" << j;
        }
        for (size_t j = take; j < k; ++j) {
            EXPECT_EQ(gpu->labels[i * k + j], -1)
                << "pad mismatch nb=" << nb << " k=" << k << " q=" << i << " pos=" << j;
        }
    }
}

TEST(MetalBfKnnParity, bf_knn_matches_cpu) {
    auto backend = make_real_metal_backend();
    if (!backend) GTEST_SKIP() << "No Metal device available";
    ASSERT_EQ(backend->device_info().backend, "metal");

    const Metric metrics[] = {Metric::euclidean, Metric::dot_product, Metric::cosine};
    const size_t dims[] = {2, 64, 128, 384};
    const size_t ns[] = {1, 1000, 10000};
    const size_t ks[] = {1, 10, 100};
    const size_t nq = 5;

    uint64_t seed = 20240717u;
    for (Metric m : metrics)
        for (size_t dim : dims)
            for (size_t n : ns)
                for (size_t k : ks)
                    run_bf_case(*backend, m, nq, n, dim, k, seed++);
}

TEST(MetalBfKnnParity, bf_knn_k_exceeds_ntotal_pads_sentinel) {
    auto backend = make_real_metal_backend();
    if (!backend) GTEST_SKIP() << "No Metal device available";
    run_bf_case(*backend, Metric::euclidean, /*nq=*/3, /*nb=*/7, /*dim=*/16, /*k=*/32, 7u);
    run_bf_case(*backend, Metric::cosine, /*nq=*/2, /*nb=*/3, /*dim=*/8, /*k=*/10, 8u);
}

// The GpuBruteForceIndex, when constructed with the REAL MetalBackend, must
// return the same (id, distance) ranking as a CPU IndexFlat over identical
// vectors -- proving the index rides real GPU kernels, not a mock.
void run_index_case(GpuBackend& be, Metric metric, size_t nb, size_t dim,
                    size_t k, uint64_t seed) {
    const bool normalize = (metric == Metric::cosine);
    auto db = make_data(nb, dim, seed, normalize);
    auto q = make_data(1, dim, seed ^ 0xD1B54A32D192ED03ULL, normalize);

    std::vector<RecordID> ids;
    ids.reserve(nb);
    for (size_t i = 0; i < nb; ++i) ids.push_back(RecordID::generate());

    GpuConfig gconfig;
    gconfig.algorithm = GpuIndexAlgorithm::BruteForce;
    GpuBruteForceIndex gpu_index(be, metric, static_cast<uint16_t>(dim), gconfig);
    auto built = gpu_index.build_from_batch(db, ids, GpuIndexBuildParams{});
    ASSERT_TRUE(built.has_value());

    IndexFlat cpu_index(metric, static_cast<uint16_t>(dim));
    for (size_t i = 0; i < nb; ++i)
        cpu_index.add_with_ids(ids[i], std::span<const float>{db.data() + i * dim, dim});

    const auto gpu_hits = gpu_index.search(std::span<const float>{q}, k);
    const auto cpu_hits = cpu_index.search(std::span<const float>{q}, k);
    ASSERT_EQ(gpu_hits.size(), cpu_hits.size());

    for (size_t j = 0; j < cpu_hits.size(); ++j) {
        if (gpu_hits[j].first != cpu_hits[j].first) {
            // fp near-tie: allowed only when the distances coincide.
            EXPECT_TRUE(near(gpu_hits[j].second, cpu_hits[j].second))
                << "non-tie id divergence metric=" << static_cast<int>(metric)
                << " nb=" << nb << " dim=" << dim << " k=" << k << " pos=" << j;
        }
        EXPECT_TRUE(near(gpu_hits[j].second, cpu_hits[j].second))
            << "distance mismatch metric=" << static_cast<int>(metric)
            << " nb=" << nb << " dim=" << dim << " k=" << k << " pos=" << j;
    }
}

TEST(MetalBfKnnParity, gpu_brute_force_index_matches_cpu_indexflat) {
    auto backend = make_real_metal_backend();
    if (!backend) GTEST_SKIP() << "No Metal device available";
    ASSERT_EQ(backend->device_info().backend, "metal");

    const Metric metrics[] = {Metric::euclidean, Metric::dot_product, Metric::cosine};
    const size_t dims[] = {2, 64, 384};
    const size_t ns[] = {1, 500, 5000};
    const size_t ks[] = {1, 10, 100};

    uint64_t seed = 31415926u;
    for (Metric m : metrics)
        for (size_t dim : dims)
            for (size_t n : ns)
                for (size_t k : ks)
                    run_index_case(*backend, m, n, dim, k, seed++);
}

}  // namespace
}  // namespace elips::gpu
