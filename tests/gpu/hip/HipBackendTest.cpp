// ELIPS HIP/ROCm backend parity test.
//
// EXPERIMENTAL — NOT HARDWARE VALIDATED. Compiled only when the HIP backend is
// enabled (-DELIPS_GPU_HIP=ON / -DELIPS_ENABLE_ROCM=ON) and skipped at runtime
// unless a real AMD GPU is present. Mirrors MetalBackendTest so that, once a
// ROCm CI runner with hardware exists, HIP gets the same CPU-parity coverage.
//
// Like the Metal test, this goes through the GpuDeviceManager / GpuSelector
// interface rather than including HipBackend.hpp directly, so it needs no HIP
// headers to build.
#include <gtest/gtest.h>

#include <span>
#include <vector>

#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/gpu_engine/GpuSelector.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu {
namespace {

// Select an available HIP backend, or nullptr if no AMD GPU is present.
std::unique_ptr<GpuPort> select_hip_backend() {
    GpuDeviceManager manager;
    auto devices = manager.probe_all_devices();
    GpuSelector selector;
    GpuConfig config;
    config.policy = GpuPolicy::Auto;
    config.preferred_backend = "hip";
    auto sel = selector.select(config, devices);
    if (sel.has_value() && *sel != nullptr &&
        (*sel)->device_info().backend == "hip") {
        return std::move(*sel);
    }
    return nullptr;
}

TEST(HipBackendTest, initialize_reports_experimental_and_non_unified) {
    auto backend = select_hip_backend();
    if (!backend) GTEST_SKIP() << "No AMD/ROCm GPU available";

    const auto info = backend->device_info();
    EXPECT_EQ(info.backend, "hip");
    EXPECT_FALSE(info.has_unified_memory);
    EXPECT_NE(info.name.find("EXPERIMENTAL"), std::string::npos);
}

TEST(HipBackendTest, allocate_upload_download_roundtrip) {
    auto backend = select_hip_backend();
    if (!backend) GTEST_SKIP() << "No AMD/ROCm GPU available";

    std::vector<float> host = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const size_t bytes = host.size() * sizeof(float);

    auto buf = backend->allocate_device(bytes);
    ASSERT_TRUE(buf.has_value());
    ASSERT_TRUE(backend->upload(host.data(), *buf, bytes).has_value());

    std::vector<float> out(host.size(), 0.0f);
    ASSERT_TRUE(backend->download(*buf, out.data(), bytes).has_value());
    for (size_t i = 0; i < host.size(); ++i) EXPECT_FLOAT_EQ(host[i], out[i]);

    backend->free_device(std::move(*buf));
}

TEST(HipBackendTest, distances_match_cpu_all_metrics) {
    auto backend = select_hip_backend();
    if (!backend) GTEST_SKIP() << "No AMD/ROCm GPU available";

    // 3 database rows, 1 query; host spans (the real caller contract).
    std::vector<float> db = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> query = {1.0f, 0.0f, 0.0f};
    const size_t nq = 1, nb = 3, dim = 3;

    for (Metric metric : {Metric::cosine, Metric::euclidean, Metric::dot_product}) {
        std::vector<float> dists(nq * nb, 0.0f);
        auto r = backend->compute_distances_batch(
            std::span<const float>{query}, std::span<const float>{db}, dists, nq,
            nb, dim, metric);
        ASSERT_TRUE(r.has_value());
        for (size_t b = 0; b < nb; ++b) {
            const float cpu = distance(
                metric, std::span<const float>{query.data(), dim},
                std::span<const float>{db.data() + b * dim, dim});
            EXPECT_NEAR(dists[b], cpu, 1e-4f) << "metric=" << static_cast<int>(metric)
                                              << " row=" << b;
        }
    }
}

TEST(HipBackendTest, top_k_selects_nearest) {
    auto backend = select_hip_backend();
    if (!backend) GTEST_SKIP() << "No AMD/ROCm GPU available";

    std::vector<float> dists = {0.5f, 0.1f, 0.9f, 0.3f};  // 1 query, nb=4
    std::vector<uint32_t> idx(2);
    std::vector<float> val(2);
    ASSERT_TRUE(backend->top_k(dists, idx, val, 1, 4, 2).has_value());

    EXPECT_EQ(idx[0], 1u);  // 0.1
    EXPECT_EQ(idx[1], 3u);  // 0.3
    EXPECT_NEAR(val[0], 0.1f, 1e-6f);
    EXPECT_NEAR(val[1], 0.3f, 1e-6f);
}

}  // namespace
}  // namespace elips::gpu
