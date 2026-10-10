#include <algorithm>

#include <gtest/gtest.h>

#include "elips/elips.hpp"
#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/gpu_engine/GpuSelector.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"

namespace elips::gpu {
namespace {

bool same_device(const GpuDeviceInfo& lhs, const GpuDeviceInfo& rhs) {
    return lhs.name == rhs.name && lhs.vendor == rhs.vendor &&
           lhs.backend == rhs.backend &&
           lhs.device_index == rhs.device_index;
}

TEST(GpuDeviceManagerTest, cpu_fallback_info_is_non_empty) {
    GpuDeviceManager manager;
    const auto info = manager.cpu_fallback_info();

    EXPECT_EQ(info.backend, "cpu");
    EXPECT_EQ(info.vendor, "CPU");
    EXPECT_FALSE(info.name.empty());
    EXPECT_TRUE(info.has_unified_memory);
}

TEST(GpuDeviceManagerTest, runtime_device_info_is_non_empty) {
    GpuDeviceManager manager;
    const auto info = manager.runtime_device_info();
    const auto devices = manager.probe_all_devices();

    EXPECT_FALSE(info.name.empty());
    EXPECT_FALSE(info.vendor.empty());
    EXPECT_FALSE(info.backend.empty());

    if (devices.empty()) {
        EXPECT_EQ(info.backend, "cpu");
        EXPECT_EQ(info.vendor, "CPU");
        return;
    }

    EXPECT_TRUE(std::any_of(devices.begin(), devices.end(),
                            [&](const GpuDeviceInfo& device) {
                                return same_device(info, device);
                            }));
}

TEST(GpuDeviceManagerTest, probe_returns_results_on_supported_system) {
    GpuDeviceManager manager;
    auto devices = manager.probe_all_devices();

#ifdef ELIPS_METAL_ENABLED
    if (devices.empty()) {
        GTEST_SKIP() << "No Metal device detected on this host.";
    }
    EXPECT_EQ(devices[0].backend, "metal");
    EXPECT_EQ(devices[0].vendor, "Apple");
    EXPECT_FALSE(devices[0].name.empty());
#else
    SUCCEED();
#endif
}

TEST(GpuDeviceManagerTest, selector_chooses_best_available) {
    GpuDeviceManager manager;
    auto devices = manager.probe_all_devices();
    GpuSelector selector;

    GpuConfig config;
    config.policy = GpuPolicy::Auto;

    auto backend = selector.select(config, devices);
    if (!devices.empty()) {
        EXPECT_TRUE(backend.has_value());
    }
}

TEST(GpuDeviceManagerTest, policy_cpu_only_returns_no_backend) {
    GpuDeviceManager manager;
    auto devices = manager.probe_all_devices();
    GpuSelector selector;

    GpuConfig config;
    config.policy = GpuPolicy::CpuOnly;

    auto backend = selector.select(config, devices);
    EXPECT_FALSE(backend.has_value());
}

TEST(GpuDeviceManagerTest, can_fit_index_estimation) {
    GpuDeviceManager manager;
    GpuDeviceInfo info;
    info.total_device_memory_bytes = 1024ULL * 1024 * 1024;
    info.free_device_memory_bytes = 512ULL * 1024 * 1024;
    info.supports_cagra = true;

    GpuConfig config;
    config.algorithm = GpuIndexAlgorithm::CagraGraph;

    EXPECT_TRUE(manager.can_fit_index(info, 10000, 128, config));
    EXPECT_TRUE(manager.can_fit_index(info, 10000, 1536, config));
}

TEST(GpuDeviceManagerTest, selector_ranks_cuda_over_hip) {
    GpuSelector selector;
    std::vector<GpuDeviceInfo> devices;

    GpuDeviceInfo cuda_dev;
    cuda_dev.backend = "cuda";
    cuda_dev.name = "Test NVIDIA";
    cuda_dev.supports_cagra = true;
    cuda_dev.total_device_memory_bytes = 1024ULL * 1024 * 1024;
    devices.push_back(cuda_dev);

    GpuDeviceInfo hip_dev;
    hip_dev.backend = "hip";
    hip_dev.name = "Test AMD";
    devices.push_back(hip_dev);

    GpuConfig config;
    config.policy = GpuPolicy::Auto;

    auto backend = selector.select(config, devices);
    if (backend.has_value()) {
        EXPECT_EQ((*backend)->device_info().vendor, "NVIDIA");
    }
}

TEST(GpuDeviceManagerTest, database_gpu_info_matches_runtime_device_info) {
    Config config;
    config.dimension(2);

    auto db = elips::open(":memory:", config);
    ASSERT_NE(db, nullptr);

    GpuDeviceManager manager;
    const auto runtime_info = manager.runtime_device_info();
    const auto db_info = db->gpu_info();

    EXPECT_EQ(db_info.name, runtime_info.name);
    EXPECT_EQ(db_info.vendor, runtime_info.vendor);
    EXPECT_EQ(db_info.backend, runtime_info.backend);
    EXPECT_EQ(db_info.device_index, runtime_info.device_index);
}

TEST(GpuDeviceManagerTest, cpu_only_policy_reports_cpu_fallback_metadata) {
    Config config;
    config.dimension(2);

    GpuConfig gpu_config;
    gpu_config.policy = GpuPolicy::CpuOnly;
    config.gpu(gpu_config);

    auto db = elips::open(":memory:", config);
    ASSERT_NE(db, nullptr);

    const auto info = db->gpu_info();
    EXPECT_EQ(info.backend, "cpu");
    EXPECT_EQ(info.vendor, "CPU");
    EXPECT_FALSE(info.name.empty());
}

}  // namespace
}  // namespace elips::gpu
