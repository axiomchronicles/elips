// ELIPS GPU Benchmark Suite
// Compares GPU-accelerated vs CPU baseline performance
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>

#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/gpu_engine/BackendSelector.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuBruteForceIndex.hpp"
#include "elips/index_engine/ExactIndex.hpp"
#include "elips/vector_engine/Metrics.hpp"

using namespace elips;
using namespace elips::gpu;
using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
    std::string name;
    double cpu_ms;
    double gpu_ms;
    double speedup;
};

BenchResult bench_search(size_t n_vectors, size_t dim, Metric metric, size_t n_queries) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> db_vecs(n_vectors * dim);
    std::vector<float> queries(n_queries * dim);
    for (auto& v : db_vecs) v = dist(rng);
    for (auto& v : queries) v = dist(rng);

    std::vector<RecordID> ids;
    for (size_t i = 0; i < n_vectors; ++i) ids.push_back(RecordID::generate());

    auto cpu_idx = std::make_unique<ExactIndex>(metric, static_cast<uint16_t>(dim));
    for (size_t i = 0; i < n_vectors; ++i) {
        cpu_idx->insert(ids[i], {db_vecs.data() + i * dim, dim});
    }

    auto cpu_start = Clock::now();
    for (size_t q = 0; q < n_queries; ++q) {
        cpu_idx->search({queries.data() + q * dim, dim}, 10);
    }
    auto cpu_end = Clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count() / n_queries;

    GpuDeviceManager manager;
    auto devices = manager.probe_all_devices();
    BackendSelector selector;
    GpuConfig config;
    config.policy = GpuPolicy::Auto;
    auto backend = selector.select(config, devices);

    double gpu_ms = 0.0;
    if (backend.has_value()) {
        GpuBruteForceIndex gpu_idx(**backend, metric, static_cast<uint16_t>(dim), config);
        gpu_idx.build_from_batch(db_vecs, ids, GpuIndexBuildParams{});

        auto gpu_start = Clock::now();
        for (size_t q = 0; q < n_queries; ++q) {
            gpu_idx.search({queries.data() + q * dim, dim}, 10);
        }
        auto gpu_end = Clock::now();
        gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count() / n_queries;
    }

    std::string dim_str = "dim=" + std::to_string(dim);
    return {dim_str, cpu_ms, gpu_ms, cpu_ms / std::max(gpu_ms, 0.001)};
}

int main() {
    std::cout << "=== ELIPS GPU Benchmark ===\n\n";
    std::cout << std::left << std::setw(12) << "Config"
              << std::setw(12) << "CPU (ms)"
              << std::setw(12) << "GPU (ms)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(48, '-') << "\n";

    for (size_t dim : {128, 384, 768, 1536}) {
        auto r = bench_search(10000, dim, Metric::cosine, 100);
        std::cout << std::left << std::setw(12) << r.name
                  << std::setw(12) << std::fixed << std::setprecision(3) << r.cpu_ms
                  << std::setw(12) << r.gpu_ms
                  << std::setw(12) << r.speedup << "x\n";
    }

    return 0;
}