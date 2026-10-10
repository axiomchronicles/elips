#ifndef ELIPS_INDEX_ENGINE_INDEX_FACTORY_HPP
#define ELIPS_INDEX_ENGINE_INDEX_FACTORY_HPP

#include <cstdint>
#include <memory>
#include <string_view>

#include "elips/Config.hpp"
#include "elips/index_engine/Index.hpp"

namespace elips {

#ifdef ELIPS_GPU_ENABLED
namespace gpu {
class GpuBackend;
}
#endif

// Builds the Index implementation selected by the configuration (DIP:
// callers depend only on Index, never a concrete index).
[[nodiscard]] std::unique_ptr<Index> make_index(const Config& config,
                                                    std::uint16_t dimension
#ifdef ELIPS_GPU_ENABLED
                                                    , gpu::GpuBackend* gpu_backend = nullptr
#endif
);

// Builds an Index from a FAISS-style factory string. Supported specs:
//   "Flat"            -> IndexFlat
//   "IVF<nlist>,Flat" -> IndexIVFFlat(nlist)        e.g. "IVF4096,Flat"
//   "IVF<nlist>,PQ<M>"-> IndexIVFPQ(nlist, M)       e.g. "IVF4096,PQ8"
//   "IVF<nlist>,SQ8"  -> IndexIVFScalarQuantizer    e.g. "IVF4096,SQ8"
//   "HNSW<M>"         -> IndexHNSW(M)               e.g. "HNSW32"
// Throws ConfigError on an unrecognized or malformed spec.
[[nodiscard]] std::unique_ptr<Index> index_factory(std::uint16_t dimension,
                                                   Metric metric,
                                                   std::string_view spec);

}  // namespace elips

#endif  // ELIPS_INDEX_ENGINE_INDEX_FACTORY_HPP
