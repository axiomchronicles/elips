#include "elips/index_engine/IndexFactory.hpp"

#include <charconv>
#include <string>

#include "elips/domain/Errors.hpp"
#include "elips/index_engine/IndexFlat.hpp"
#include "elips/index_engine/IndexHNSW.hpp"
#include "elips/index_engine/IndexIVFFlat.hpp"
#include "elips/index_engine/IndexIVFPQ.hpp"
#include "elips/index_engine/IndexIVFScalarQuantizer.hpp"

#ifdef ELIPS_GPU_ENABLED
#include "elips/gpu_engine/GpuBruteForceIndex.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuHybridIndex.hpp"
#endif

namespace elips {
namespace {

std::unique_ptr<Index> make_cpu_index(const Config& config,
                                          std::uint16_t dimension) {
    // A quantizer is only handed over once one has been trained; until then the
    // index stores full fp32 even when a codec is configured.
    const quant::QuantizerPtr& quantizer = config.quantizer();
    switch (config.index()) {
        case IndexType::exact:
            return std::make_unique<IndexFlat>(config.metric(), dimension,
                                                quantizer);
        case IndexType::graph:
            return std::make_unique<IndexHNSW>(
                config.metric(), dimension, config.graph_params(), quantizer);
    }
    return std::make_unique<IndexFlat>(config.metric(), dimension, quantizer);
}

// Parses the unsigned integer that `spec` carries after `prefix`, up to `end`
// (one-past-the-last digit, or npos for the rest of the string). Throws on a
// missing/garbage number.
std::size_t parse_uint(std::string_view spec, std::string_view prefix,
                       std::size_t end) {
    const std::string_view digits =
        spec.substr(prefix.size(),
                    end == std::string_view::npos ? end : end - prefix.size());
    std::size_t value = 0;
    const char* first = digits.data();
    const char* last = first + digits.size();
    const auto [ptr, ec] = std::from_chars(first, last, value);
    if (ec != std::errc{} || ptr != last || digits.empty()) {
        throw ConfigError{"index_factory: malformed count in '" +
                          std::string(spec) + "'"};
    }
    return value;
}

}  // namespace

std::unique_ptr<Index> index_factory(std::uint16_t dimension, Metric metric,
                                     std::string_view spec) {
    if (spec == "Flat") {
        return std::make_unique<IndexFlat>(metric, dimension);
    }
    if (spec.starts_with("HNSW")) {
        const std::size_t m = parse_uint(spec, "HNSW", std::string_view::npos);
        HnswParams params;
        params.max_connections = m;
        return std::make_unique<IndexHNSW>(metric, dimension, params);
    }
    if (spec.starts_with("IVF")) {
        const std::size_t comma = spec.find(',');
        if (comma == std::string_view::npos) {
            throw ConfigError{"index_factory: IVF spec needs a ',<coarse>' "
                              "suffix, got '" +
                              std::string(spec) + "'"};
        }
        const std::size_t nlist = parse_uint(spec, "IVF", comma);
        const std::string_view sub = spec.substr(comma + 1);
        if (sub == "Flat") {
            return std::make_unique<IndexIVFFlat>(metric, dimension, nlist);
        }
        if (sub == "SQ8") {
            return std::make_unique<IndexIVFScalarQuantizer>(metric, dimension,
                                                             nlist);
        }
        if (sub.starts_with("PQ")) {
            // "IVF<nlist>,PQ<M>" -> IVFPQ with pq_dim=M, 8-bit codes. The ctor
            // rejects an M that does not divide the dimension.
            const std::size_t m = parse_uint(sub, "PQ", std::string_view::npos);
            return std::make_unique<IndexIVFPQ>(
                metric, dimension, nlist, static_cast<std::uint32_t>(m));
        }
        throw ConfigError{"index_factory: IVF supports 'Flat', 'PQ<M>', 'SQ8' "
                          "coarse suffixes, got '" +
                          std::string(spec) + "'"};
    }
    throw ConfigError{"index_factory: unknown spec '" + std::string(spec) + "'"};
}

std::unique_ptr<Index> make_index(const Config& config,
                                      std::uint16_t dimension
#ifdef ELIPS_GPU_ENABLED
                                      , gpu::GpuBackend* gpu_backend
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
