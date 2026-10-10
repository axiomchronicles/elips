// Real-Metal parity test for the Flat path: exercises the actual Metal
// distance GEMM + GPU threadgroup top-k on this machine (NOT RecordingBackend)
// and checks label/distance parity against a CPU IndexFlat-style oracle.
//
// Skips only if no Metal device is present. On Apple Silicon it must run.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include "elips/gpu_engine/BackendSelector.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu {
namespace {

constexpr float kRelTol = 6e-3f;  // per _faiss_3I/3H fp32 relative tolerance

// A real Metal backend obtained via the production selector (never the test
// RecordingBackend). Returns nullptr if no Metal device is available.
std::unique_ptr<GpuBackend> make_real_metal_backend() {
    GpuDeviceManager manager;
    BackendSelector selector;
    GpuConfig config;
    config.policy = GpuPolicy::Auto;
    auto devices = manager.probe_all_devices();
    auto sel = selector.select(config, devices);
    if (!sel.has_value()) return nullptr;
    auto backend = std::move(*sel);
    if (backend->device_info().backend != "metal") return nullptr;  // not real Metal
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

// CPU oracle: top-k by (distance, index) ascending — matches the retired CPU
// top_k selection with smaller-index tiebreak.
std::vector<uint32_t> cpu_topk(const std::vector<float>& row, size_t nb, size_t k) {
    std::vector<uint32_t> idx(nb);
    std::iota(idx.begin(), idx.end(), 0u);
    size_t take = std::min(k, nb);
    std::partial_sort(idx.begin(), idx.begin() + take, idx.end(),
                      [&](uint32_t a, uint32_t b) {
                          if (row[a] != row[b]) return row[a] < row[b];
                          return a < b;
                      });
    idx.resize(take);
    return idx;
}

bool near(float a, float b) {
    return std::fabs(a - b) <= kRelTol * (1.0f + std::fabs(b));
}

void run_case(GpuBackend& be, Metric metric, size_t nq, size_t nb, size_t dim,
              size_t k, uint64_t seed) {
    const bool normalize = (metric == Metric::cosine);
    auto db = make_data(nb, dim, seed, normalize);
    auto q = make_data(nq, dim, seed ^ 0x9E3779B97F4A7C15ULL, normalize);

    // --- GPU distance matrix ---
    std::vector<float> d_gpu(nq * nb);
    auto r = be.compute_distances_batch(std::span<const float>{q}, std::span<const float>{db},
                                        d_gpu, nq, nb, dim, metric);
    ASSERT_TRUE(r.has_value());

    // --- CPU distance matrix (oracle) + distance parity ---
    std::vector<float> d_cpu(nq * nb);
    for (size_t i = 0; i < nq; ++i)
        for (size_t j = 0; j < nb; ++j)
            d_cpu[i * nb + j] = distance(
                metric, std::span<const float>{q.data() + i * dim, dim},
                std::span<const float>{db.data() + j * dim, dim});

    for (size_t t = 0; t < nq * nb; ++t)
        ASSERT_TRUE(near(d_gpu[t], d_cpu[t]))
            << "dist mismatch metric=" << static_cast<int>(metric) << " dim=" << dim
            << " nb=" << nb << " at " << t << " gpu=" << d_gpu[t] << " cpu=" << d_cpu[t];

    // --- GPU top-k ---
    std::vector<uint32_t> gpu_idx(nq * k);
    std::vector<float> gpu_val(nq * k);
    auto rk = be.top_k(d_gpu, gpu_idx, gpu_val, nq, nb, k);
    ASSERT_TRUE(rk.has_value());

    const size_t take = std::min(k, nb);
    for (size_t i = 0; i < nq; ++i) {
        std::vector<float> row(d_gpu.begin() + i * nb, d_gpu.begin() + (i + 1) * nb);

        // (1) Exact top_k-kernel parity: CPU selection on the SAME (GPU) distances
        //     must match the GPU kernel bit-for-bit (deterministic tiebreak).
        auto exact = cpu_topk(row, nb, k);
        for (size_t j = 0; j < take; ++j)
            EXPECT_EQ(gpu_idx[i * k + j], exact[j])
                << "topk label mismatch metric=" << static_cast<int>(metric)
                << " nb=" << nb << " k=" << k << " q=" << i << " pos=" << j;

        // (2) End-to-end pipeline parity vs the CPU IndexFlat oracle (CPU
        //     distances + CPU selection). Labels must match except for genuine
        //     fp near-ties at the k-boundary.
        std::vector<float> row_cpu(d_cpu.begin() + i * nb, d_cpu.begin() + (i + 1) * nb);
        auto oracle = cpu_topk(row_cpu, nb, k);
        for (size_t j = 0; j < take; ++j) {
            if (gpu_idx[i * k + j] != oracle[j]) {
                EXPECT_TRUE(near(row_cpu[gpu_idx[i * k + j]], row_cpu[oracle[j]]))
                    << "non-tie label divergence metric=" << static_cast<int>(metric)
                    << " nb=" << nb << " k=" << k << " q=" << i << " pos=" << j;
            }
            EXPECT_TRUE(near(gpu_val[i * k + j], row_cpu[oracle[j]]))
                << "topk value mismatch metric=" << static_cast<int>(metric)
                << " nb=" << nb << " k=" << k << " q=" << i << " pos=" << j;
        }

        // (3) k > ntotal: positions >= nb are padded with sentinel index (-1).
        for (size_t j = take; j < k; ++j)
            EXPECT_EQ(gpu_idx[i * k + j], static_cast<uint32_t>(-1))
                << "pad mismatch nb=" << nb << " k=" << k << " q=" << i << " pos=" << j;
    }
}

TEST(MetalRealKernelParity, flat_distance_and_topk_match_cpu) {
    auto backend = make_real_metal_backend();
    if (!backend) GTEST_SKIP() << "No Metal device available";
    ASSERT_EQ(backend->device_info().backend, "metal");

    const Metric metrics[] = {Metric::euclidean, Metric::dot_product, Metric::cosine};
    const size_t dims[] = {2, 64, 128, 384};
    const size_t ns[] = {1, 1000, 10000};
    const size_t ks[] = {1, 10, 100};
    const size_t nq = 5;

    uint64_t seed = 1234567u;
    for (Metric m : metrics)
        for (size_t dim : dims)
            for (size_t n : ns)
                for (size_t k : ks)
                    run_case(*backend, m, nq, n, dim, k, seed++);
}

// Explicit k > ntotal padding case (n small, k large).
TEST(MetalRealKernelParity, k_exceeds_ntotal_pads_sentinel) {
    auto backend = make_real_metal_backend();
    if (!backend) GTEST_SKIP() << "No Metal device available";
    run_case(*backend, Metric::euclidean, /*nq=*/3, /*nb=*/7, /*dim=*/16, /*k=*/32, 99u);
    run_case(*backend, Metric::cosine, /*nq=*/2, /*nb=*/3, /*dim=*/8, /*k=*/10, 100u);
}

}  // namespace
}  // namespace elips::gpu
