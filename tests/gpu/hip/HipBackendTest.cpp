// ============================================================================
// HIP/ROCm backend parity test  —  EXPERIMENTAL / NOT HARDWARE-VALIDATED
// ----------------------------------------------------------------------------
// Compiled only when ELIPS_HIP_ENABLED. Every case skips (GTEST_SKIP) unless a
// real AMD device is present, so it is safe to build in CI without a GPU and is
// exercised only on ROCm hardware runners. Mirrors the CUDA parity test
// (tests/gpu/cuda/CudaBackendTest.cpp) and the Metal parity test.
// ============================================================================
#ifdef ELIPS_HIP_ENABLED

#include <gtest/gtest.h>

#include <span>
#include <vector>

#include "elips/gpu_engine/GpuBackend.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu {
namespace {

// Selects the HIP backend specifically (preferred_backend == "hip"), or returns
// nullptr when no AMD device is available.
std::unique_ptr<GpuBackend> select_hip() {
    GpuDeviceManager manager;
    auto devices = manager.probe_all_devices();
    GpuConfig config;
    config.policy = GpuPolicy::Auto;
    config.preferred_backend = "hip";
    auto sel = manager.select(config, devices);
    if (sel.has_value() && *sel) return std::move(*sel);
    return nullptr;
}

class HipBackendTest : public ::testing::Test {
protected:
    void SetUp() override { backend_ = select_hip(); }
    std::unique_ptr<GpuBackend> backend_;
};

TEST_F(HipBackendTest, initialize_reports_experimental_hip_device) {
    if (!backend_) GTEST_SKIP() << "No AMD/ROCm device";
    EXPECT_TRUE(backend_->is_available());
    const auto info = backend_->device_info();
    EXPECT_EQ(info.backend, "hip");
    EXPECT_FALSE(info.has_unified_memory);
    EXPECT_NE(info.name.find("EXPERIMENTAL"), std::string::npos);
}

TEST_F(HipBackendTest, allocate_upload_download_roundtrip) {
    if (!backend_) GTEST_SKIP() << "No AMD/ROCm device";
    std::vector<float> host = {1, 2, 3, 4, 5, 6, 7, 8};
    const size_t bytes = host.size() * sizeof(float);

    auto buf = backend_->allocate_device(bytes);
    ASSERT_TRUE(buf.has_value());
    ASSERT_TRUE(backend_->upload(host.data(), *buf, bytes).has_value());

    std::vector<float> out(host.size(), 0.0f);
    ASSERT_TRUE(backend_->download(*buf, out.data(), bytes).has_value());
    for (size_t i = 0; i < host.size(); ++i) EXPECT_FLOAT_EQ(host[i], out[i]);
    backend_->free_device(std::move(*buf));
}

TEST_F(HipBackendTest, distance_and_topk_match_cpu) {
    if (!backend_) GTEST_SKIP() << "No AMD/ROCm device";
    // 3 normalized basis vectors; query == first basis vector.
    std::vector<float> db = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<float> query = {1, 0, 0};
    const size_t nq = 1, nb = 3, dim = 3;

    for (Metric metric : {Metric::cosine, Metric::euclidean, Metric::dot_product}) {
        std::vector<float> dists(nq * nb);
        auto d = backend_->compute_distances_batch(query, db, dists, nq, nb, dim, metric);
        ASSERT_TRUE(d.has_value());
        for (size_t j = 0; j < nb; ++j) {
            float cpu = distance(metric, std::span<const float>{query.data(), dim},
                                 std::span<const float>{db.data() + j * dim, dim});
            EXPECT_NEAR(dists[j], cpu, 1e-4f) << "metric " << static_cast<int>(metric);
        }

        std::vector<uint32_t> idx(nb);
        std::vector<float> vals(nb);
        ASSERT_TRUE(backend_->top_k(dists, idx, vals, nq, nb, nb).has_value());
        EXPECT_EQ(idx[0], 0u); // nearest to query is db[0]
        for (size_t j = 1; j < nb; ++j) EXPECT_LE(vals[j - 1], vals[j]); // ascending
    }
}

} // namespace
} // namespace elips::gpu

#endif // ELIPS_HIP_ENABLED
