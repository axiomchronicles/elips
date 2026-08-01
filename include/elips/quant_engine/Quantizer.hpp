#ifndef ELIPS_QUANT_ENGINE_QUANTIZER_HPP
#define ELIPS_QUANT_ENGINE_QUANTIZER_HPP

#include <cstddef>
#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>
#include <span>
#include <string_view>
#include <vector>

#include "elips/Config.hpp"
#include "elips/quant_engine/QuantParams.hpp"

namespace elips::quant {

// A trained codec that compresses a fp32 vector into a fixed-width byte code.
//
// A quantizer is immutable once trained: every method is const and safe to call
// concurrently from any number of threads. Vaults and indexes share one via
// std::shared_ptr<const Quantizer> rather than each holding a copy of the
// codebook, which for PQ at 8 bits is 256 * dimension floats.
//
// Distance is computed asymmetrically (ADC): the query stays at full precision
// and only the stored side is compressed. make_lut() does the O(dimension) work
// once per query; lut_distance() is then O(code_bytes) per candidate, with no
// decode and no allocation, which is what makes a compressed scan faster than
// the fp32 scan it replaces rather than merely smaller.
class Quantizer {
public:
    Quantizer() = default;
    virtual ~Quantizer() = default;
    Quantizer(const Quantizer&) = delete;
    Quantizer& operator=(const Quantizer&) = delete;
    Quantizer(Quantizer&&) = delete;
    Quantizer& operator=(Quantizer&&) = delete;

    [[nodiscard]] virtual CodecId codec() const noexcept = 0;
    [[nodiscard]] virtual std::size_t code_bytes() const noexcept = 0;
    [[nodiscard]] virtual std::uint16_t dimension() const noexcept = 0;
    // The metric this codec was trained against. A quantizer trained for one
    // metric produces wrong distances under another, so it is fixed at training
    // time rather than passed per query.
    [[nodiscard]] virtual Metric metric() const noexcept = 0;

    virtual void encode(std::span<const float> vector,
                        std::span<std::uint8_t> code_out) const = 0;
    virtual void decode(std::span<const std::uint8_t> code,
                        std::span<float> vector_out) const = 0;

    // Query-side precomputation. The result is opaque; pass it back to
    // lut_distance() unchanged.
    [[nodiscard]] virtual std::vector<float> make_lut(
        std::span<const float> query) const = 0;
    [[nodiscard]] virtual float lut_distance(
        std::span<const float> lut,
        std::span<const std::uint8_t> code) const noexcept = 0;

    // Writes the full codec state. The first byte is the CodecId, which is what
    // load() dispatches on.
    virtual void serialize(std::ostream& out) const = 0;

    // Convenience wrappers for callers that hold a single vector.
    [[nodiscard]] std::vector<std::uint8_t> encode(
        std::span<const float> vector) const {
        std::vector<std::uint8_t> code(code_bytes());
        encode(vector, code);
        return code;
    }
    [[nodiscard]] std::vector<float> decode(
        std::span<const std::uint8_t> code) const {
        std::vector<float> vector(dimension());
        decode(code, vector);
        return vector;
    }
};

using QuantizerPtr = std::shared_ptr<const Quantizer>;  // declared in QuantParams.hpp

// Trains a codec over `n` row-major vectors of `dimension` floats.
//
// Throws ConfigError when the parameters cannot describe a valid codec (codec
// none, pq_bits outside [4, 8], a pq_dim that does not divide the dimension) or
// when the training set is empty. Callers with more rows than they want to
// spend on training should sample before calling; see train_sample_cap.
[[nodiscard]] QuantizerPtr train(const QuantParams& params, Metric metric,
                                 std::uint16_t dimension,
                                 std::span<const float> rows, std::size_t n);

// Reads a codec written by Quantizer::serialize(). Throws StorageError on a
// truncated, malformed, or unknown-codec stream.
[[nodiscard]] QuantizerPtr load(std::istream& in);

// Rows beyond this count add training time without meaningfully improving the
// codebook, so callers sample down to it. Codebook quality is a function of
// coverage, not of dataset size.
inline constexpr std::size_t train_sample_cap = 100'000;

// Largest divisor of `dimension` at or below dimension/8, which is the usual
// default subspace count. Returns 1 when nothing else divides.
[[nodiscard]] std::uint32_t choose_pq_dim(std::uint16_t dimension,
                                          std::uint32_t configured) noexcept;

// Validates `params` against a dimension without training. Throws ConfigError
// with a specific message; used by Config and the CLI to reject bad input
// before any data is touched.
void validate(const QuantParams& params, std::uint16_t dimension);

// Bytes one vector occupies under `params` at `dimension`, for reporting a
// compression ratio before training. Zero when the codec is none.
[[nodiscard]] std::size_t code_bytes_for(const QuantParams& params,
                                         std::uint16_t dimension);

}  // namespace elips::quant

#endif  // ELIPS_QUANT_ENGINE_QUANTIZER_HPP
