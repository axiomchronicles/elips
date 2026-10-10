#include <exception>
#include <iostream>

#include "elips/elips.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuDeviceManager.hpp"

int main() {
    try {
        elips::gpu::GpuDeviceManager manager;
        if (manager.probe_all_devices().empty()) {
            std::cout << "No GPU available; skipping lifecycle probe.\n";
            return 0;
        }

        elips::Config config;
        config.dimension(2).metric(elips::Metric::cosine);

        elips::gpu::GpuConfig gpu_config;
        gpu_config.policy = elips::gpu::GpuPolicy::RequireGpu;
        gpu_config.index_build_mode =
            elips::gpu::IndexBuildMode::GpuBuild_GpuServe;
        gpu_config.algorithm = elips::gpu::GpuIndexAlgorithm::BruteForce;
        config.gpu(gpu_config);

        {
            auto db = elips::open(":memory:", config);
            const auto info = db->gpu_info();
            if (info.backend == "cpu" || info.backend.empty()) {
                std::cerr << "Expected a live GPU backend, got '" << info.backend
                          << "'.\n";
                return 1;
            }

            auto& collection = db->collection("docs");
            collection.add(elips::Vector{{1.0F, 0.0F}});
            collection.add(elips::Vector{{0.0F, 1.0F}});

            const auto hits = collection.search(elips::Vector{{1.0F, 0.0F}}, 2);
            if (hits.size() != 2U) {
                std::cerr << "Unexpected hit count: " << hits.size() << '\n';
                return 1;
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Lifecycle probe failed: " << ex.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "Lifecycle probe failed with unknown exception.\n";
        return 1;
    }
}
