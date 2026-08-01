#include "elips/quant_engine/Quantizer.hpp"

#include <cstring>

#include "elips/domain/Errors.hpp"
#include "elips/quant_engine/ProductQuantizer.hpp"
#include "elips/quant_engine/ScalarQuantizer.hpp"
#include "elips/quant_engine/detail/KMeans.hpp"
#include "elips/storage/Serialization.hpp"

namespace elips::quant {

const char* to_string(CodecId codec) noexcept {
    switch (codec) {
        case CodecId::none:
            return "none";
        case CodecId::pq:
            return "pq";
        case CodecId::opq:
            return "opq";
        case CodecId::sq8:
            return "sq8";
    }
    return "none";
}

CodecId codec_from_string(const char* name) {
    if (name == nullptr) {
        throw ConfigError{"unknown quantization codec name"};
    }
    if (std::strcmp(name, "none") == 0) {
        return CodecId::none;
    }
    if (std::strcmp(name, "pq") == 0) {
        return CodecId::pq;
    }
    if (std::strcmp(name, "opq") == 0) {
        return CodecId::opq;
    }
    if (std::strcmp(name, "sq8") == 0 || std::strcmp(name, "sq") == 0) {
        return CodecId::sq8;
    }
    throw ConfigError{"unknown quantization codec name"};
}

std::uint32_t choose_pq_dim(std::uint16_t dimension,
                            std::uint32_t configured) noexcept {
    if (dimension == 0) {
        return 1;
    }
    if (configured > 0 && dimension % configured == 0) {
        return configured;
    }
    // Aim for 8 components per subspace, then walk down to the nearest divisor
    // so the subspaces partition the vector exactly.
    std::uint32_t guess = std::max<std::uint32_t>(1, dimension / 8);
    while (guess > 1 && dimension % guess != 0) {
        --guess;
    }
    return guess;
}

void validate(const QuantParams& params, std::uint16_t dimension) {
    if (params.codec == CodecId::none) {
        return;
    }
    if (dimension == 0) {
        throw ConfigError{"quantization requires a non-zero dimension"};
    }
    if (params.codec == CodecId::sq8) {
        return;  // no subspace or codebook constraints
    }
    if (params.pq_bits < 4 || params.pq_bits > 8) {
        throw ConfigError{"pq_bits must be between 4 and 8"};
    }
    if (params.pq_dim != 0 && dimension % params.pq_dim != 0) {
        throw ConfigError{"pq_dim must divide the vector dimension"};
    }
    const std::uint32_t effective = choose_pq_dim(dimension, params.pq_dim);
    if (effective == 0 || dimension % effective != 0) {
        throw ConfigError{"no valid subspace count for this dimension"};
    }
}

std::size_t code_bytes_for(const QuantParams& params, std::uint16_t dimension) {
    switch (params.codec) {
        case CodecId::none:
            return 0;
        case CodecId::sq8:
            return dimension;
        case CodecId::pq:
        case CodecId::opq:
            return choose_pq_dim(dimension, params.pq_dim);
    }
    return 0;
}

QuantizerPtr train(const QuantParams& params, Metric metric,
                   std::uint16_t dimension, std::span<const float> rows,
                   std::size_t n) {
    validate(params, dimension);

    if (params.codec == CodecId::none) {
        throw ConfigError{"cannot train a quantizer with codec none"};
    }
    if (n == 0) {
        throw ConfigError{"quantizer training requires at least one vector"};
    }
    if (rows.size() < n * static_cast<std::size_t>(dimension)) {
        throw InvalidVector{"training data is smaller than n * dimension"};
    }

    if (params.codec == CodecId::sq8) {
        return ScalarQuantizer::train(metric, dimension, rows, n);
    }

    const std::uint32_t pq_dim = choose_pq_dim(dimension, params.pq_dim);
    const std::uint32_t opq_iters =
        (params.codec == CodecId::opq) ? std::max<std::uint32_t>(1, params.opq_iters)
                                       : 0U;
    return ProductQuantizer::train(metric, dimension, pq_dim, params.pq_bits,
                                   std::max<std::uint32_t>(1, params.train_iters),
                                   opq_iters, rows, n);
}

QuantizerPtr load(std::istream& in) {
    const auto raw = elips::detail::get<std::uint8_t>(in);
    if (!in) {
        throw StorageError{"truncated quantizer"};
    }
    switch (static_cast<CodecId>(raw)) {
        case CodecId::pq:
            return ProductQuantizer::deserialize(in, CodecId::pq);
        case CodecId::opq:
            return ProductQuantizer::deserialize(in, CodecId::opq);
        case CodecId::sq8:
            return ScalarQuantizer::deserialize(in);
        case CodecId::none:
            break;
    }
    throw StorageError{"unknown quantizer codec tag"};
}

}  // namespace elips::quant
