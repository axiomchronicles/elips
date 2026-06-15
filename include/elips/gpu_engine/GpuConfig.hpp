#ifndef ELIPS_GPU_ENGINE_GPU_CONFIG_HPP
#define ELIPS_GPU_ENGINE_GPU_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

namespace elips::gpu {

enum class GpuPolicy {
    Auto,
    PreferGpu,
    RequireGpu,
    CpuOnly,
    Specific,
};

enum class IndexBuildMode {
    GpuBuild_CpuServe,
    GpuBuild_GpuServe,
    Hybrid,
};

enum class GpuIndexAlgorithm {
    Auto,
    CagraGraph,
    IvfFlat,
    IvfPq,
    BruteForce,
};

enum class GpuPrecision { FP32, FP16, Int8, Auto };

struct GraphIndexBuildParams {
    size_t intermediate_graph_degree{128};
    size_t graph_degree{64};
    enum class BuildAlgo { IvfPq, NnDescent, IterativeSearch } build_algo{BuildAlgo::IvfPq};
    size_t nn_descent_iterations{20};
    float compression_ratio{0.0f};
};

struct IvfPqBuildParams {
    size_t n_lists{1024};
    uint32_t pq_dim{0};
    uint32_t pq_bits{8};
    bool add_data_on_build{true};
    size_t kmeans_n_iters{20};
    float kmeans_trainset_fraction{0.5f};
};

struct GpuIndexBuildParams {
    std::variant<GraphIndexBuildParams, IvfPqBuildParams> params{GraphIndexBuildParams{}};
};

struct GpuConfig {
    GpuPolicy policy{GpuPolicy::Auto};

    std::string preferred_backend;
    int32_t device_index{-1};

    IndexBuildMode index_build_mode{IndexBuildMode::GpuBuild_GpuServe};

    GpuIndexAlgorithm algorithm{GpuIndexAlgorithm::Auto};

    size_t device_memory_pool_bytes{0};
    size_t pinned_host_pool_bytes{256UL * 1024 * 1024};
    bool use_unified_memory{false};

    size_t default_ef_search_gpu{64};
    size_t dynamic_batch_window_us{500};
    size_t dynamic_batch_max_size{256};
    bool enable_fp16_search{false};

    GraphIndexBuildParams graph_params;
    IvfPqBuildParams ivf_pq_params;

    GpuPrecision search_precision{GpuPrecision::Auto};

    bool auto_rebuild_on_startup{false};
    float rebuild_threshold_ratio{0.1f};

    bool enable_profiling{false};
    bool emit_kernel_timings{false};
};

} // namespace elips::gpu

#endif