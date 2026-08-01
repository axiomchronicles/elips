#include "elips/quant_engine/ProductQuantizer.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "elips/domain/Errors.hpp"
#include "elips/quant_engine/detail/KMeans.hpp"
#include "elips/storage/Serialization.hpp"

namespace elips::quant {
namespace {

using detail::apply_matrix;
using detail::apply_matrix_transposed;

// PQ's ADC table holds, per subspace, the distance from the query's subvector to
// every centroid in that subspace. Summing pq_dim table lookups then yields the
// squared L2 distance between the query and the reconstruction, without ever
// materializing the reconstruction.
//
// Cosine and dot_product are handled by tabulating the negated partial dot
// product instead, since both are monotone functions of dot(q, x): cosine over
// L2-normalized inputs is 1 - dot, and dot_product is -dot. Storing the negated
// partial keeps lut_distance() a plain sum for all three metrics.
constexpr std::size_t max_ksub = 256;

}  // namespace

ProductQuantizer::ProductQuantizer(Metric metric, std::uint16_t dimension,
                                   std::uint32_t pq_dim, std::uint32_t pq_bits,
                                   std::vector<float> codebook,
                                   std::vector<float> rotation)
    : metric_(metric),
      dimension_(dimension),
      pq_dim_(pq_dim),
      pq_bits_(pq_bits),
      codebook_(std::move(codebook)),
      rotation_(std::move(rotation)) {}

void ProductQuantizer::rotate(std::span<const float> in,
                              std::span<float> out) const noexcept {
    if (rotation_.empty()) {
        std::copy_n(in.data(), dimension_, out.data());
        return;
    }
    apply_matrix(rotation_, in, out, dimension_);
}

void ProductQuantizer::unrotate(std::span<const float> in,
                                std::span<float> out) const noexcept {
    if (rotation_.empty()) {
        std::copy_n(in.data(), dimension_, out.data());
        return;
    }
    apply_matrix_transposed(rotation_, in, out, dimension_);
}

void ProductQuantizer::encode_rotated(
    std::span<const float> rotated, std::span<std::uint8_t> code_out) const noexcept {
    const std::size_t sub = sub_dim();
    const std::size_t k = ksub();
    for (std::size_t m = 0; m < pq_dim_; ++m) {
        code_out[m] = detail::nearest_centroid(
            rotated.subspan(m * sub, sub),
            std::span<const float>{codebook_}.subspan(m * k * sub, k * sub), sub,
            k);
    }
}

void ProductQuantizer::encode(std::span<const float> vector,
                              std::span<std::uint8_t> code_out) const {
    if (vector.size() != dimension_) {
        throw DimensionMismatch{"vector dimension does not match quantizer"};
    }
    if (code_out.size() < code_bytes()) {
        throw InvalidVector{"code buffer is smaller than the code width"};
    }
    if (rotation_.empty()) {
        encode_rotated(vector, code_out);
        return;
    }
    std::vector<float> rotated(dimension_);
    rotate(vector, rotated);
    encode_rotated(rotated, code_out);
}

void ProductQuantizer::decode(std::span<const std::uint8_t> code,
                              std::span<float> vector_out) const {
    if (code.size() < code_bytes()) {
        throw InvalidVector{"code is shorter than the code width"};
    }
    if (vector_out.size() < dimension_) {
        throw DimensionMismatch{"output buffer is smaller than the dimension"};
    }

    const std::size_t sub = sub_dim();
    const std::size_t k = ksub();

    if (rotation_.empty()) {
        for (std::size_t m = 0; m < pq_dim_; ++m) {
            const std::size_t centroid = code[m];
            std::copy_n(codebook_.data() + (((m * k) + centroid) * sub), sub,
                        vector_out.data() + (m * sub));
        }
        return;
    }

    std::vector<float> rotated(dimension_);
    for (std::size_t m = 0; m < pq_dim_; ++m) {
        const std::size_t centroid = code[m];
        std::copy_n(codebook_.data() + (((m * k) + centroid) * sub), sub,
                    rotated.data() + (m * sub));
    }
    unrotate(rotated, vector_out);
}

std::vector<float> ProductQuantizer::make_lut(
    std::span<const float> query) const {
    if (query.size() != dimension_) {
        throw DimensionMismatch{"query dimension does not match quantizer"};
    }

    const std::size_t sub = sub_dim();
    const std::size_t k = ksub();

    std::vector<float> rotated_storage;
    std::span<const float> effective = query;
    if (!rotation_.empty()) {
        rotated_storage.resize(dimension_);
        rotate(query, rotated_storage);
        effective = rotated_storage;
    }

    std::vector<float> lut(static_cast<std::size_t>(pq_dim_) * k);
    const bool use_dot = metric_ != Metric::euclidean;

    for (std::size_t m = 0; m < pq_dim_; ++m) {
        const float* query_sub = effective.data() + (m * sub);
        const float* codebook = codebook_.data() + (m * k * sub);
        float* out = lut.data() + (m * k);
        for (std::size_t c = 0; c < k; ++c) {
            const float* centroid = codebook + (c * sub);
            float accumulator = 0.0F;
            if (use_dot) {
                for (std::size_t i = 0; i < sub; ++i) {
                    accumulator += query_sub[i] * centroid[i];
                }
                out[c] = -accumulator;  // negated so summing sorts ascending
            } else {
                for (std::size_t i = 0; i < sub; ++i) {
                    const float d = query_sub[i] - centroid[i];
                    accumulator += d * d;
                }
                out[c] = accumulator;
            }
        }
    }
    return lut;
}

float ProductQuantizer::lut_distance(
    std::span<const float> lut,
    std::span<const std::uint8_t> code) const noexcept {
    const std::size_t k = ksub();
    float sum = 0.0F;
    for (std::size_t m = 0; m < pq_dim_; ++m) {
        sum += lut[(m * k) + code[m]];
    }
    switch (metric_) {
        case Metric::cosine:
            // lut already holds -dot, so 1 - dot is 1 + sum.
            return 1.0F + sum;
        case Metric::dot_product:
            return sum;
        case Metric::euclidean:
            // The table accumulates squared L2; the scalar kernel in Metrics.cpp
            // returns the root, so match it or the two disagree on threshold
            // comparisons.
            return std::sqrt(std::max(0.0F, sum));
    }
    return sum;
}

void ProductQuantizer::serialize(std::ostream& out) const {
    elips::detail::put<std::uint8_t>(out, static_cast<std::uint8_t>(codec()));
    elips::detail::put<std::uint8_t>(out, static_cast<std::uint8_t>(metric_));
    elips::detail::put<std::uint16_t>(out, dimension_);
    elips::detail::put<std::uint32_t>(out, pq_dim_);
    elips::detail::put<std::uint32_t>(out, pq_bits_);
    elips::detail::put<std::uint32_t>(out,
                               static_cast<std::uint32_t>(codebook_.size()));
    out.write(reinterpret_cast<const char*>(codebook_.data()),
              static_cast<std::streamsize>(codebook_.size() * sizeof(float)));
    elips::detail::put<std::uint32_t>(out,
                               static_cast<std::uint32_t>(rotation_.size()));
    if (!rotation_.empty()) {
        out.write(reinterpret_cast<const char*>(rotation_.data()),
                  static_cast<std::streamsize>(rotation_.size() * sizeof(float)));
    }
}

std::unique_ptr<ProductQuantizer> ProductQuantizer::deserialize(
    std::istream& in, CodecId codec) {
    const auto metric = static_cast<Metric>(elips::detail::get<std::uint8_t>(in));
    const auto dimension = elips::detail::get<std::uint16_t>(in);
    const auto pq_dim = elips::detail::get<std::uint32_t>(in);
    const auto pq_bits = elips::detail::get<std::uint32_t>(in);

    if (dimension == 0 || pq_dim == 0 || dimension % pq_dim != 0 ||
        pq_bits < 4 || pq_bits > 8) {
        throw StorageError{"corrupt product quantizer header"};
    }

    const auto codebook_floats = elips::detail::get<std::uint32_t>(in);
    const std::size_t expected_codebook =
        static_cast<std::size_t>(pq_dim) * (std::size_t{1} << pq_bits) *
        (dimension / pq_dim);
    if (codebook_floats != expected_codebook) {
        throw StorageError{"product quantizer codebook size mismatch"};
    }
    elips::detail::check_length(in, static_cast<std::uint64_t>(codebook_floats) *
                                 sizeof(float));
    std::vector<float> codebook(codebook_floats);
    in.read(reinterpret_cast<char*>(codebook.data()),
            static_cast<std::streamsize>(codebook_floats * sizeof(float)));

    const auto rotation_floats = elips::detail::get<std::uint32_t>(in);
    std::vector<float> rotation;
    if (rotation_floats != 0) {
        const std::size_t expected_rotation =
            static_cast<std::size_t>(dimension) * dimension;
        if (rotation_floats != expected_rotation) {
            throw StorageError{"product quantizer rotation size mismatch"};
        }
        elips::detail::check_length(in, static_cast<std::uint64_t>(rotation_floats) *
                                     sizeof(float));
        rotation.resize(rotation_floats);
        in.read(reinterpret_cast<char*>(rotation.data()),
                static_cast<std::streamsize>(rotation_floats * sizeof(float)));
    }

    if (!in) {
        throw StorageError{"truncated product quantizer"};
    }
    if ((codec == CodecId::opq) != !rotation.empty()) {
        throw StorageError{"product quantizer codec does not match its rotation"};
    }

    return std::make_unique<ProductQuantizer>(metric, dimension, pq_dim, pq_bits,
                                              std::move(codebook),
                                              std::move(rotation));
}

std::unique_ptr<ProductQuantizer> ProductQuantizer::train(
    Metric metric, std::uint16_t dimension, std::uint32_t pq_dim,
    std::uint32_t pq_bits, std::uint32_t train_iters, std::uint32_t opq_iters,
    std::span<const float> rows, std::size_t n) {
    const std::size_t sub = dimension / pq_dim;
    const std::size_t k = std::size_t{1} << pq_bits;
    if (k > max_ksub) {
        throw ConfigError{"pq_bits above 8 would not fit a byte-wide code"};
    }

    std::vector<float> codebook(static_cast<std::size_t>(pq_dim) * k * sub);
    std::vector<float> rotation;

    if (opq_iters == 0) {
        detail::train_pq_codebook(rows, n, dimension, pq_dim, k, train_iters,
                                  codebook);
        return std::make_unique<ProductQuantizer>(metric, dimension, pq_dim,
                                                  pq_bits, std::move(codebook),
                                                  std::vector<float>{});
    }

    // OPQ alternates two steps: fit a codebook in the current rotated basis,
    // then re-fit the rotation that best maps the raw data onto the resulting
    // reconstructions. Each step is a non-increasing move on the same
    // objective, so the loop converges monotonically.
    rotation = detail::identity_matrix(dimension);
    std::vector<float> rotated(n * dimension);
    std::vector<float> reconstructed(n * dimension);
    std::vector<std::uint8_t> codes(n * pq_dim);

    for (std::uint32_t iter = 0; iter < opq_iters; ++iter) {
        for (std::size_t row = 0; row < n; ++row) {
            apply_matrix(rotation, rows.subspan(row * dimension, dimension),
                         std::span<float>{rotated}.subspan(row * dimension,
                                                           dimension),
                         dimension);
        }

        detail::train_pq_codebook(rotated, n, dimension, pq_dim, k, train_iters,
                                  codebook);
        detail::encode_pq(rotated, codebook, n, dimension, pq_dim, k, codes);

        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t m = 0; m < pq_dim; ++m) {
                const std::size_t centroid = codes[(row * pq_dim) + m];
                std::copy_n(codebook.data() + (((m * k) + centroid) * sub), sub,
                            reconstructed.data() + (row * dimension) + (m * sub));
            }
        }

        // The last iteration's codebook is the one that ships, so re-solving
        // the rotation after it would leave the two out of step.
        if (iter + 1 < opq_iters) {
            rotation = detail::solve_procrustes(rows, reconstructed, n, dimension);
        }
    }

    return std::make_unique<ProductQuantizer>(metric, dimension, pq_dim, pq_bits,
                                              std::move(codebook),
                                              std::move(rotation));
}

}  // namespace elips::quant
