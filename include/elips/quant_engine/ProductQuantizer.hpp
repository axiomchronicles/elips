#ifndef ELIPS_QUANT_ENGINE_PRODUCT_QUANTIZER_HPP
#define ELIPS_QUANT_ENGINE_PRODUCT_QUANTIZER_HPP

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <span>
#include <vector>

#include "elips/quant_engine/Quantizer.hpp"

namespace elips::quant {

// Product quantization: the vector is split into `pq_dim` contiguous subspaces
// of `sub_dim` components each, and every subspace is coded independently
// against its own k-means codebook of `ksub = 2^pq_bits` centroids. One vector
// becomes `pq_dim` bytes regardless of dimension, so at dim 128 with 16
// subspaces the compression ratio is 32x.
//
// This class serves both PQ and OPQ. OPQ prepends a learned dim x dim
// orthonormal rotation that redistributes variance evenly across subspaces
// before coding, which is what recovers the recall PQ loses when the dimensions
// are correlated (the common case for learned embeddings). `rotation_` empty
// means plain PQ; populated means OPQ. Sharing one class is not a shortcut: the
// two differ only by that pre-multiply, so a second class would duplicate the
// codebook, the encoder, and the ADC path to add one matrix multiply.
class ProductQuantizer final : public Quantizer {
public:
    // Trains from row-major data. `n` must be non-zero and `rows` must hold at
    // least n * dimension floats. `opq_iters` of zero yields plain PQ.
    static std::unique_ptr<ProductQuantizer> train(
        Metric metric, std::uint16_t dimension, std::uint32_t pq_dim,
        std::uint32_t pq_bits, std::uint32_t train_iters, std::uint32_t opq_iters,
        std::span<const float> rows, std::size_t n);

    // Reads the body written by serialize(), with the leading CodecId byte
    // already consumed by quant::load().
    static std::unique_ptr<ProductQuantizer> deserialize(std::istream& in,
                                                         CodecId codec);

    [[nodiscard]] CodecId codec() const noexcept override {
        return rotation_.empty() ? CodecId::pq : CodecId::opq;
    }
    [[nodiscard]] std::size_t code_bytes() const noexcept override {
        return pq_dim_;
    }
    [[nodiscard]] std::uint16_t dimension() const noexcept override {
        return dimension_;
    }
    [[nodiscard]] Metric metric() const noexcept override { return metric_; }

    void encode(std::span<const float> vector,
                std::span<std::uint8_t> code_out) const override;
    void decode(std::span<const std::uint8_t> code,
                std::span<float> vector_out) const override;

    [[nodiscard]] std::vector<float> make_lut(
        std::span<const float> query) const override;
    [[nodiscard]] float lut_distance(
        std::span<const float> lut,
        std::span<const std::uint8_t> code) const noexcept override;

    void serialize(std::ostream& out) const override;

    // Exposed for IndexSnapshot transfer and tests.
    [[nodiscard]] const std::vector<float>& codebook() const noexcept {
        return codebook_;
    }
    [[nodiscard]] const std::vector<float>& rotation() const noexcept {
        return rotation_;
    }
    [[nodiscard]] std::uint32_t pq_dim() const noexcept { return pq_dim_; }
    [[nodiscard]] std::uint32_t pq_bits() const noexcept { return pq_bits_; }

    ProductQuantizer(Metric metric, std::uint16_t dimension, std::uint32_t pq_dim,
                     std::uint32_t pq_bits, std::vector<float> codebook,
                     std::vector<float> rotation);

private:
    // Applies rotation_ to `in`, writing dimension_ floats to `out`. A no-op
    // copy when the rotation is empty.
    void rotate(std::span<const float> in, std::span<float> out) const noexcept;
    // Applies the transpose, which is the inverse because rotation_ is
    // orthonormal.
    void unrotate(std::span<const float> in, std::span<float> out) const noexcept;
    // Codes `rotated` (already in the rotated basis) into code_out.
    void encode_rotated(std::span<const float> rotated,
                        std::span<std::uint8_t> code_out) const noexcept;

    [[nodiscard]] std::size_t sub_dim() const noexcept {
        return static_cast<std::size_t>(dimension_) / pq_dim_;
    }
    [[nodiscard]] std::size_t ksub() const noexcept {
        return std::size_t{1} << pq_bits_;
    }

    Metric metric_;
    std::uint16_t dimension_;
    std::uint32_t pq_dim_;
    std::uint32_t pq_bits_;
    // pq_dim_ * ksub() * sub_dim() floats: subspace-major, then centroid, then
    // component.
    std::vector<float> codebook_;
    // dimension_ * dimension_ floats, row-major. Empty for plain PQ.
    std::vector<float> rotation_;
};

}  // namespace elips::quant

#endif  // ELIPS_QUANT_ENGINE_PRODUCT_QUANTIZER_HPP
