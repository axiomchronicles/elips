#ifndef ELIPS_QUANT_ENGINE_SCALAR_QUANTIZER_HPP
#define ELIPS_QUANT_ENGINE_SCALAR_QUANTIZER_HPP

#include <cstddef>
#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>
#include <span>
#include <vector>

#include "elips/quant_engine/Quantizer.hpp"

namespace elips::quant {

// Per-dimension int8 scalar quantization: each component is linearly mapped
// from its observed [min, max] range onto 0-255. One vector becomes `dimension`
// bytes, a flat 4x compression, and reconstruction error is bounded by half a
// quantization step per component rather than by codebook coverage.
//
// SQ trades PQ's compression ratio for near-exact recall and a trivial training
// pass (one min/max sweep, no iteration). It is the right default when memory
// is merely a concern rather than the binding constraint, and the safe choice
// when the data distribution is unknown, since it has no codebook to be
// unrepresentative of anything.
class ScalarQuantizer final : public Quantizer {
public:
    // Sweeps `n` row-major vectors for per-dimension bounds. A dimension with
    // zero range gets a unit scale so decode returns the constant exactly
    // instead of dividing by zero.
    static std::unique_ptr<ScalarQuantizer> train(Metric metric,
                                                  std::uint16_t dimension,
                                                  std::span<const float> rows,
                                                  std::size_t n);

    static std::unique_ptr<ScalarQuantizer> deserialize(std::istream& in);

    [[nodiscard]] CodecId codec() const noexcept override {
        return CodecId::sq8;
    }
    [[nodiscard]] std::size_t code_bytes() const noexcept override {
        return dimension_;
    }
    [[nodiscard]] std::uint16_t dimension() const noexcept override {
        return dimension_;
    }
    [[nodiscard]] Metric metric() const noexcept override { return metric_; }

    void encode(std::span<const float> vector,
                std::span<std::uint8_t> code_out) const override;
    void decode(std::span<const std::uint8_t> code,
                std::span<float> vector_out) const override;

    // SQ has no per-subspace table to precompute, so the "lut" is a 256-entry
    // per-dimension expansion of the query's contribution. That keeps
    // lut_distance() free of the multiply-and-scale work and makes it a pure
    // table gather, matching the PQ path's cost model.
    [[nodiscard]] std::vector<float> make_lut(
        std::span<const float> query) const override;
    [[nodiscard]] float lut_distance(
        std::span<const float> lut,
        std::span<const std::uint8_t> code) const noexcept override;

    void serialize(std::ostream& out) const override;

    [[nodiscard]] const std::vector<float>& mins() const noexcept {
        return mins_;
    }
    [[nodiscard]] const std::vector<float>& scales() const noexcept {
        return scales_;
    }

    ScalarQuantizer(Metric metric, std::uint16_t dimension,
                    std::vector<float> mins, std::vector<float> scales);

private:
    Metric metric_;
    std::uint16_t dimension_;
    std::vector<float> mins_;    // dimension_ entries
    std::vector<float> scales_;  // dimension_ entries: (max - min) / 255
};

}  // namespace elips::quant

#endif  // ELIPS_QUANT_ENGINE_SCALAR_QUANTIZER_HPP
