#include "elips/index_engine/IndexFactory.hpp"

#include "elips/index_engine/ExactIndex.hpp"
#include "elips/index_engine/HierarchicalGraphIndex.hpp"

#ifdef ELIPS_GPU_ENABLED
#include "elips/gpu_engine/GpuBruteForceIndex.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuHybridIndex.hpp"
#endif

namespace elips {
namespace {

std::unique_ptr<IndexPort> make_cpu_index(const Config& config,
                                          std::uint16_t dimension) {
    // A quantizer is only handed over once one has been trained; until then the
    // index stores full fp32 even when a codec is configured.
    const quant::QuantizerPtr& quantizer = config.quantizer();
    switch (config.index()) {
        case IndexType::exact:
            return std::make_unique<ExactIndex>(config.metric(), dimension,
                                                quantizer);
        case IndexType::graph:
            return std::make_unique<HierarchicalGraphIndex>(
                config.metric(), dimension, config.graph_params(), quantizer);
    }
    return std::make_unique<ExactIndex>(config.metric(), dimension, quantizer);
}

}  // namespace

std::unique_ptr<IndexPort> make_index(const Config& config,
                                      std::uint16_t dimension
#ifdef ELIPS_GPU_ENABLED
                                      , gpu::GpuPort* gpu_backend
#endif
) {
#ifdef ELIPS_GPU_ENABLED
    if (gpu_backend != nullptr && config.has_gpu() &&
        config.gpu().policy != gpu::GpuPolicy::CpuOnly) {
        if (config.gpu().index_build_mode == gpu::IndexBuildMode::GpuBuild_GpuServe &&
            config.gpu().algorithm == gpu::GpuIndexAlgorithm::BruteForce) {
            return std::make_unique<gpu::GpuBruteForceIndex>(
                *gpu_backend, config.metric(), dimension, config.gpu());
        }
        return std::make_unique<gpu::GpuHybridIndex>(
            *gpu_backend, make_cpu_index(config, dimension), config.metric(),
            dimension, config.gpu());
    }
#endif
    return make_cpu_index(config, dimension);
}

}  // namespace elips
