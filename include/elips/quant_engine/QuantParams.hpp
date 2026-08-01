#ifndef ELIPS_QUANT_ENGINE_QUANT_PARAMS_HPP
#define ELIPS_QUANT_ENGINE_QUANT_PARAMS_HPP

#include <cstdint>
#include <memory>

// Lightweight quantization configuration, split out of Quantizer.hpp so that
// Config.hpp can carry it without pulling in Metric (which Config.hpp itself
// defines) and forming an include cycle.
namespace elips::quant {

class Quantizer;

// A trained codec is immutable and shared: vaults and indexes observe the same
// instance rather than each copying a codebook that, for 8-bit PQ, is 256
// centroids per subspace.
using QuantizerPtr = std::shared_ptr<const Quantizer>;

// On-disk and in-memory codec discriminator. Values are persisted in snapshots,
// segments, and the WAL: never renumber, only append.
enum class CodecId : std::uint8_t {
    none = 0,  // full fp32, no compression
    pq = 1,    // product quantization
    opq = 2,   // product quantization with a learned rotation
    sq8 = 3,   // per-dimension int8 scalar quantization
};

[[nodiscard]] const char* to_string(CodecId codec) noexcept;
[[nodiscard]] CodecId codec_from_string(const char* name);

// Tunable parameters for the quantizers. Mirrors GraphParams: a plain struct of
// defaults that a caller overrides selectively.
struct QuantParams {
    CodecId codec{CodecId::none};
    // Number of subspaces (PQ/OPQ). 0 selects the largest divisor of the
    // dimension at or below dimension/8.
    std::uint32_t pq_dim{0};
    // Bits per subquantizer code. Bounded to [4, 8]; the codebook holds
    // 2^pq_bits centroids per subspace.
    std::uint32_t pq_bits{8};
    // Lloyd iterations per subspace codebook.
    std::uint32_t train_iters{10};
    // Rotation/codebook alternations. Ignored unless codec is opq.
    std::uint32_t opq_iters{4};
    // Reserved: fraction of vectors to retain at full precision for exact
    // reranking. Not implemented; see ADR-0011.
    float refine_fraction{0.0F};
};

}  // namespace elips::quant

#endif  // ELIPS_QUANT_ENGINE_QUANT_PARAMS_HPP
