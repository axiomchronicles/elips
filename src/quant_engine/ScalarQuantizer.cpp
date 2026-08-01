#include "elips/quant_engine/ScalarQuantizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "elips/domain/Errors.hpp"
#include "elips/storage/Serialization.hpp"

namespace elips::quant {
namespace {

constexpr std::size_t levels = 256;
constexpr float max_level = 255.0F;

}  // namespace

ScalarQuantizer::ScalarQuantizer(Metric metric, std::uint16_t dimension,
                                 std::vector<float> mins,
                                 std::vector<float> scales)
    : metric_(metric),
      dimension_(dimension),
      mins_(std::move(mins)),
      scales_(std::move(scales)) {}

std::unique_ptr<ScalarQuantizer> ScalarQuantizer::train(
    Metric metric, std::uint16_t dimension, std::span<const float> rows,
    std::size_t n) {
    std::vector<float> mins(dimension, std::numeric_limits<float>::max());
    std::vector<float> maxes(dimension, std::numeric_limits<float>::lowest());

    for (std::size_t row = 0; row < n; ++row) {
        const float* vector = rows.data() + (row * dimension);
        for (std::size_t i = 0; i < dimension; ++i) {
            mins[i] = std::min(mins[i], vector[i]);
            maxes[i] = std::max(maxes[i], vector[i]);
        }
    }

    std::vector<float> scales(dimension);
    for (std::size_t i = 0; i < dimension; ++i) {
        const float range = maxes[i] - mins[i];
        // A constant dimension has no range to divide by. A unit scale makes
        // encode() emit level 0 and decode() return the constant exactly.
        scales[i] = (range > 0.0F) ? (range / max_level) : 1.0F;
    }

    return std::make_unique<ScalarQuantizer>(metric, dimension, std::move(mins),
                                             std::move(scales));
}

void ScalarQuantizer::encode(std::span<const float> vector,
                             std::span<std::uint8_t> code_out) const {
    if (vector.size() != dimension_) {
        throw DimensionMismatch{"vector dimension does not match quantizer"};
    }
    if (code_out.size() < code_bytes()) {
        throw InvalidVector{"code buffer is smaller than the code width"};
    }
    for (std::size_t i = 0; i < dimension_; ++i) {
        // Clamp rather than wrap: a post-training insert may fall outside the
        // observed range, and wrapping would map an extreme value onto the
        // opposite end of the scale.
        const float level = (vector[i] - mins_[i]) / scales_[i];
        const float clamped = std::clamp(level, 0.0F, max_level);
        code_out[i] = static_cast<std::uint8_t>(std::lround(clamped));
    }
}

void ScalarQuantizer::decode(std::span<const std::uint8_t> code,
                             std::span<float> vector_out) const {
    if (code.size() < code_bytes()) {
        throw InvalidVector{"code is shorter than the code width"};
    }
    if (vector_out.size() < dimension_) {
        throw DimensionMismatch{"output buffer is smaller than the dimension"};
    }
    for (std::size_t i = 0; i < dimension_; ++i) {
        vector_out[i] = mins_[i] + (static_cast<float>(code[i]) * scales_[i]);
    }
}

std::vector<float> ScalarQuantizer::make_lut(std::span<const float> query) const {
    if (query.size() != dimension_) {
        throw DimensionMismatch{"query dimension does not match quantizer"};
    }

    // dimension_ * 256 entries: the full per-dimension contribution for every
    // possible byte value. Costs one pass at query time and turns the per
    // candidate work into a pure gather-and-add.
    std::vector<float> lut(static_cast<std::size_t>(dimension_) * levels);
    const bool use_dot = metric_ != Metric::euclidean;

    for (std::size_t i = 0; i < dimension_; ++i) {
        float* out = lut.data() + (i * levels);
        const float base = mins_[i];
        const float scale = scales_[i];
        const float q = query[i];
        for (std::size_t level = 0; level < levels; ++level) {
            const float value = base + (static_cast<float>(level) * scale);
            if (use_dot) {
                out[level] = -(q * value);  // negated so summing sorts ascending
            } else {
                const float d = q - value;
                out[level] = d * d;
            }
        }
    }
    return lut;
}

float ScalarQuantizer::lut_distance(
    std::span<const float> lut,
    std::span<const std::uint8_t> code) const noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < dimension_; ++i) {
        sum += lut[(i * levels) + code[i]];
    }
    switch (metric_) {
        case Metric::cosine:
            return 1.0F + sum;
        case Metric::dot_product:
            return sum;
        case Metric::euclidean:
            return std::sqrt(std::max(0.0F, sum));
    }
    return sum;
}

void ScalarQuantizer::serialize(std::ostream& out) const {
    elips::detail::put<std::uint8_t>(out, static_cast<std::uint8_t>(CodecId::sq8));
    elips::detail::put<std::uint8_t>(out, static_cast<std::uint8_t>(metric_));
    elips::detail::put<std::uint16_t>(out, dimension_);
    out.write(reinterpret_cast<const char*>(mins_.data()),
              static_cast<std::streamsize>(mins_.size() * sizeof(float)));
    out.write(reinterpret_cast<const char*>(scales_.data()),
              static_cast<std::streamsize>(scales_.size() * sizeof(float)));
}

std::unique_ptr<ScalarQuantizer> ScalarQuantizer::deserialize(std::istream& in) {
    const auto metric = static_cast<Metric>(elips::detail::get<std::uint8_t>(in));
    const auto dimension = elips::detail::get<std::uint16_t>(in);
    if (dimension == 0) {
        throw StorageError{"corrupt scalar quantizer header"};
    }

    const std::uint64_t bytes_needed =
        static_cast<std::uint64_t>(dimension) * sizeof(float) * 2ULL;
    elips::detail::check_length(in, bytes_needed);

    std::vector<float> mins(dimension);
    std::vector<float> scales(dimension);
    in.read(reinterpret_cast<char*>(mins.data()),
            static_cast<std::streamsize>(dimension * sizeof(float)));
    in.read(reinterpret_cast<char*>(scales.data()),
            static_cast<std::streamsize>(dimension * sizeof(float)));
    if (!in) {
        throw StorageError{"truncated scalar quantizer"};
    }

    return std::make_unique<ScalarQuantizer>(metric, dimension, std::move(mins),
                                             std::move(scales));
}

}  // namespace elips::quant
