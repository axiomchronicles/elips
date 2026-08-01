// Unit tests for the backend-free quantization codecs (elips::quant).
//
// Covers the invariants every codec must hold regardless of how it compresses:
// a code is the advertised width, decode inverts encode within the codec's
// error budget, ADC agrees with the scalar distance kernel over the decoded
// vector, and a serialize/load round trip is bit-exact.
#include <cmath>
#include <cstdint>
#include <random>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include "elips/domain/Errors.hpp"
#include "elips/quant_engine/ProductQuantizer.hpp"
#include "elips/quant_engine/Quantizer.hpp"
#include "elips/quant_engine/ScalarQuantizer.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace {

using elips::Metric;
using elips::quant::CodecId;
using elips::quant::QuantParams;

constexpr std::uint16_t kDim = 64;
constexpr std::size_t kRows = 2000;

// Independent gaussian columns: the case PQ handles well, since no rotation can
// improve on subspaces that are already uncorrelated.
std::vector<float> independent_dataset(std::size_t n, std::uint16_t dim,
                                       unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> rows(n * dim);
    for (float& v : rows) {
        v = dist(rng);
    }
    return rows;
}

// Strongly correlated columns with wildly unequal per-subspace variance. Plain
// PQ wastes codebook capacity here because the leading subspace carries most of
// the energy; this is the structure OPQ's rotation exists to fix.
std::vector<float> correlated_dataset(std::size_t n, std::uint16_t dim,
                                      unsigned seed = 11) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> rows(n * dim);
    for (std::size_t row = 0; row < n; ++row) {
        float* out = rows.data() + (row * dim);
        const float shared = dist(rng);
        for (std::size_t i = 0; i < dim; ++i) {
            // Variance decays sharply with the column index and every column
            // shares a common factor.
            const float scale = 1.0F / static_cast<float>(1 + i);
            out[i] = (shared * scale * 4.0F) + (dist(rng) * scale);
        }
    }
    return rows;
}

double mean_squared_error(const elips::quant::Quantizer& codec,
                          const std::vector<float>& rows, std::size_t n,
                          std::uint16_t dim) {
    std::vector<std::uint8_t> code(codec.code_bytes());
    std::vector<float> decoded(dim);
    double total = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
        const std::span<const float> original{rows.data() + (row * dim), dim};
        codec.encode(original, code);
        codec.decode(code, decoded);
        for (std::size_t i = 0; i < dim; ++i) {
            const double d = static_cast<double>(original[i]) - decoded[i];
            total += d * d;
        }
    }
    return total / static_cast<double>(n * dim);
}

QuantParams params_for(CodecId codec) {
    QuantParams params;
    params.codec = codec;
    params.pq_dim = 16;
    params.pq_bits = 8;
    params.train_iters = 8;
    params.opq_iters = 3;
    return params;
}

}  // namespace

// ---------------------------- code width -----------------------------------

TEST(QuantizerTest, CodeWidthMatchesTheAdvertisedCompressionRatio) {
    const auto rows = independent_dataset(kRows, kDim);

    const auto pq = elips::quant::train(params_for(CodecId::pq), Metric::euclidean,
                                        kDim, rows, kRows);
    // 16 subspaces at one byte each, against 64 floats.
    EXPECT_EQ(pq->code_bytes(), 16U);
    EXPECT_EQ(kDim * sizeof(float) / pq->code_bytes(), 16U);

    const auto sq = elips::quant::train(params_for(CodecId::sq8), Metric::euclidean,
                                        kDim, rows, kRows);
    EXPECT_EQ(sq->code_bytes(), kDim);
    EXPECT_EQ(kDim * sizeof(float) / sq->code_bytes(), 4U);
}

TEST(QuantizerTest, AutoPqDimPicksADivisorOfTheDimension) {
    // 0 means "choose for me"; the result must still partition the vector.
    QuantParams params = params_for(CodecId::pq);
    params.pq_dim = 0;
    const auto rows = independent_dataset(500, kDim);
    const auto pq =
        elips::quant::train(params, Metric::euclidean, kDim, rows, 500);
    EXPECT_EQ(kDim % pq->code_bytes(), 0U);
}

// -------------------------- encode / decode --------------------------------

TEST(QuantizerTest, ProductQuantizerDecodeApproximatesTheOriginal) {
    const auto rows = independent_dataset(kRows, kDim);
    const auto pq = elips::quant::train(params_for(CodecId::pq), Metric::euclidean,
                                        kDim, rows, kRows);
    // Unit-variance input; a 16x8-bit code should land far below the variance
    // it is approximating, or the codebook is not learning anything.
    EXPECT_LT(mean_squared_error(*pq, rows, kRows, kDim), 0.35);
}

TEST(QuantizerTest, ScalarQuantizerErrorIsBoundedByHalfAStep) {
    const auto rows = independent_dataset(kRows, kDim);
    const auto sq = elips::quant::train(params_for(CodecId::sq8), Metric::euclidean,
                                        kDim, rows, kRows);

    // SQ's error is analytic: no component can be off by more than half a
    // quantization step of its own dimension.
    std::vector<std::uint8_t> code(sq->code_bytes());
    std::vector<float> decoded(kDim);
    const auto& scales =
        dynamic_cast<const elips::quant::ScalarQuantizer&>(*sq).scales();

    for (std::size_t row = 0; row < 200; ++row) {
        const std::span<const float> original{rows.data() + (row * kDim), kDim};
        sq->encode(original, code);
        sq->decode(code, decoded);
        for (std::size_t i = 0; i < kDim; ++i) {
            EXPECT_LE(std::fabs(original[i] - decoded[i]),
                      (scales[i] * 0.5F) + 1e-5F)
                << "dimension " << i;
        }
    }
}

TEST(QuantizerTest, ScalarQuantizerHandlesAConstantDimension) {
    // A zero-range dimension has no step to divide by; it must decode exactly
    // rather than produce NaN.
    std::vector<float> rows(100 * kDim, 0.0F);
    for (std::size_t row = 0; row < 100; ++row) {
        rows[(row * kDim)] = 3.5F;  // dimension 0 is constant
        rows[(row * kDim) + 1] = static_cast<float>(row);
    }
    const auto sq = elips::quant::train(params_for(CodecId::sq8),
                                        Metric::euclidean, kDim, rows, 100);
    std::vector<float> decoded(kDim);
    const auto code = sq->encode(std::span<const float>{rows.data(), kDim});
    sq->decode(code, decoded);
    EXPECT_FLOAT_EQ(decoded[0], 3.5F);
    for (const float v : decoded) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(QuantizerTest, EncodeClampsValuesOutsideTheTrainedRange) {
    // A post-training insert can exceed the observed bounds. It must saturate,
    // not wrap around to the opposite end of the scale.
    const auto rows = independent_dataset(500, kDim);
    const auto sq = elips::quant::train(params_for(CodecId::sq8),
                                        Metric::euclidean, kDim, rows, 500);

    std::vector<float> extreme(kDim, 1000.0F);
    const auto high = sq->encode(extreme);
    std::vector<float> tiny(kDim, -1000.0F);
    const auto low = sq->encode(tiny);

    for (std::size_t i = 0; i < kDim; ++i) {
        EXPECT_EQ(high[i], 255U);
        EXPECT_EQ(low[i], 0U);
    }
}

// ------------------------------- ADC ---------------------------------------

TEST(QuantizerTest, LutDistanceMatchesTheScalarKernelOverTheDecodedVector) {
    // The whole point of ADC is to skip the decode. It is only correct if it
    // agrees with what the decode would have produced.
    const auto rows = independent_dataset(kRows, kDim);
    const auto queries = independent_dataset(20, kDim, 99);

    for (const Metric metric :
         {Metric::euclidean, Metric::cosine, Metric::dot_product}) {
        for (const CodecId codec : {CodecId::pq, CodecId::opq, CodecId::sq8}) {
            const auto quantizer =
                elips::quant::train(params_for(codec), metric, kDim, rows, kRows);

            for (std::size_t q = 0; q < 20; ++q) {
                const std::span<const float> query{queries.data() + (q * kDim),
                                                   kDim};
                const auto lut = quantizer->make_lut(query);

                for (std::size_t row = 0; row < 25; ++row) {
                    const auto code = quantizer->encode(
                        std::span<const float>{rows.data() + (row * kDim), kDim});
                    const auto decoded = quantizer->decode(code);
                    const float expected = elips::distance(metric, query, decoded);
                    const float actual = quantizer->lut_distance(lut, code);
                    EXPECT_NEAR(actual, expected, 1e-3F)
                        << "codec " << elips::quant::to_string(codec)
                        << " metric " << static_cast<int>(metric);
                }
            }
        }
    }
}

TEST(QuantizerTest, LutDistanceOrdersCandidatesLikeTheExactKernel) {
    // Ordering matters more than absolute values: the index sorts on these.
    const auto rows = independent_dataset(kRows, kDim);
    const auto sq = elips::quant::train(params_for(CodecId::sq8),
                                        Metric::euclidean, kDim, rows, kRows);
    const std::span<const float> query{rows.data(), kDim};
    const auto lut = sq->make_lut(query);

    for (std::size_t row = 1; row < 100; ++row) {
        const auto code = sq->encode(
            std::span<const float>{rows.data() + (row * kDim), kDim});
        // Every other row must rank strictly farther than the query's own row.
        const auto self = sq->encode(query);
        EXPECT_LE(sq->lut_distance(lut, self), sq->lut_distance(lut, code));
    }
}

// ---------------------------- serialization ---------------------------------

TEST(QuantizerTest, SerializeRoundTripReproducesIdenticalCodes) {
    const auto rows = independent_dataset(kRows, kDim);
    const auto queries = independent_dataset(5, kDim, 3);

    for (const CodecId codec : {CodecId::pq, CodecId::opq, CodecId::sq8}) {
        const auto original =
            elips::quant::train(params_for(codec), Metric::cosine, kDim, rows, kRows);

        std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
        original->serialize(buffer);
        const auto restored = elips::quant::load(buffer);

        ASSERT_EQ(restored->codec(), original->codec());
        ASSERT_EQ(restored->code_bytes(), original->code_bytes());
        ASSERT_EQ(restored->dimension(), original->dimension());
        ASSERT_EQ(restored->metric(), original->metric());

        for (std::size_t row = 0; row < 200; ++row) {
            const std::span<const float> vector{rows.data() + (row * kDim), kDim};
            EXPECT_EQ(restored->encode(vector), original->encode(vector))
                << "codec " << elips::quant::to_string(codec) << " row " << row;
        }

        // Distances must survive the round trip too, not just the codes.
        const std::span<const float> query{queries.data(), kDim};
        const auto code = original->encode(query);
        EXPECT_FLOAT_EQ(restored->lut_distance(restored->make_lut(query), code),
                        original->lut_distance(original->make_lut(query), code));
    }
}

TEST(QuantizerTest, LoadRejectsTruncatedAndGarbageInput) {
    const auto rows = independent_dataset(500, kDim);
    const auto pq = elips::quant::train(params_for(CodecId::pq),
                                        Metric::euclidean, kDim, rows, 500);

    std::stringstream full(std::ios::in | std::ios::out | std::ios::binary);
    pq->serialize(full);
    const std::string bytes = full.str();

    // Truncated at every 64th offset: each must fail, none may hang or
    // over-allocate on an unvalidated length prefix.
    for (std::size_t cut = 1; cut < bytes.size(); cut += 64) {
        std::stringstream truncated(bytes.substr(0, cut),
                                    std::ios::in | std::ios::binary);
        EXPECT_ANY_THROW((void)elips::quant::load(truncated)) << "cut at " << cut;
    }

    std::stringstream empty(std::ios::in | std::ios::binary);
    EXPECT_THROW((void)elips::quant::load(empty), elips::StorageError);

    std::stringstream unknown(std::string("\x7f", 1),
                              std::ios::in | std::ios::binary);
    EXPECT_THROW((void)elips::quant::load(unknown), elips::StorageError);
}

// ------------------------------ validation ----------------------------------

TEST(QuantizerTest, TrainRejectsInvalidParameters) {
    const auto rows = independent_dataset(100, kDim);

    QuantParams none;
    EXPECT_THROW(
        (void)elips::quant::train(none, Metric::cosine, kDim, rows, 100),
        elips::ConfigError);

    QuantParams bad_bits = params_for(CodecId::pq);
    bad_bits.pq_bits = 3;
    EXPECT_THROW(
        (void)elips::quant::train(bad_bits, Metric::cosine, kDim, rows, 100),
        elips::ConfigError);
    bad_bits.pq_bits = 9;
    EXPECT_THROW(
        (void)elips::quant::train(bad_bits, Metric::cosine, kDim, rows, 100),
        elips::ConfigError);

    QuantParams indivisible = params_for(CodecId::pq);
    indivisible.pq_dim = 7;  // 64 % 7 != 0
    EXPECT_THROW(
        (void)elips::quant::train(indivisible, Metric::cosine, kDim, rows, 100),
        elips::ConfigError);

    EXPECT_THROW((void)elips::quant::train(params_for(CodecId::pq),
                                           Metric::cosine, kDim, rows, 0),
                 elips::ConfigError);

    // Fewer floats than n * dimension claims.
    const std::vector<float> short_rows(kDim, 0.0F);
    EXPECT_THROW((void)elips::quant::train(params_for(CodecId::pq),
                                           Metric::cosine, kDim, short_rows, 100),
                 elips::InvalidVector);
}

TEST(QuantizerTest, EncodeAndDecodeRejectMismatchedBuffers) {
    const auto rows = independent_dataset(200, kDim);
    const auto pq = elips::quant::train(params_for(CodecId::pq),
                                        Metric::euclidean, kDim, rows, 200);

    const std::vector<float> wrong_dimension(kDim + 1, 0.0F);
    std::vector<std::uint8_t> code(pq->code_bytes());
    EXPECT_THROW(pq->encode(wrong_dimension, code), elips::DimensionMismatch);
    EXPECT_THROW((void)pq->make_lut(wrong_dimension), elips::DimensionMismatch);

    std::vector<std::uint8_t> undersized(pq->code_bytes() - 1);
    EXPECT_THROW(
        pq->encode(std::span<const float>{rows.data(), kDim}, undersized),
        elips::InvalidVector);

    std::vector<float> small_output(kDim - 1);
    EXPECT_THROW(pq->decode(code, small_output), elips::DimensionMismatch);
}

// --------------------------------- OPQ --------------------------------------

TEST(QuantizerTest, OpqBeatsPlainPqOnCorrelatedDimensions) {
    // The reason OPQ exists: when variance is concentrated in a few directions,
    // fixed contiguous subspaces split the vector badly and most of the codebook
    // is spent on subspaces carrying almost no energy. A learned rotation
    // redistributes that variance before the split.
    const auto rows = correlated_dataset(kRows, kDim);

    const auto pq = elips::quant::train(params_for(CodecId::pq), Metric::euclidean,
                                        kDim, rows, kRows);
    const auto opq = elips::quant::train(params_for(CodecId::opq),
                                         Metric::euclidean, kDim, rows, kRows);

    const double pq_error = mean_squared_error(*pq, rows, kRows, kDim);
    const double opq_error = mean_squared_error(*opq, rows, kRows, kDim);

    EXPECT_LE(opq_error, pq_error)
        << "opq " << opq_error << " vs pq " << pq_error;
    // Same code width, so the gain is pure recall rather than bought with bytes.
    EXPECT_EQ(opq->code_bytes(), pq->code_bytes());
}

TEST(QuantizerTest, OpqRotationIsOrthonormal) {
    // Decode applies the transpose as the inverse. That is only correct if the
    // trained matrix is actually orthonormal; a drifting one silently corrupts
    // every reconstruction.
    const auto rows = correlated_dataset(kRows, kDim);
    const auto opq = elips::quant::train(params_for(CodecId::opq),
                                         Metric::euclidean, kDim, rows, kRows);
    const auto& rotation =
        dynamic_cast<const elips::quant::ProductQuantizer&>(*opq).rotation();
    ASSERT_EQ(rotation.size(), static_cast<std::size_t>(kDim) * kDim);

    // R * R^T must be the identity.
    for (std::size_t i = 0; i < kDim; ++i) {
        for (std::size_t j = 0; j < kDim; ++j) {
            float dot = 0.0F;
            for (std::size_t k = 0; k < kDim; ++k) {
                dot += rotation[(i * kDim) + k] * rotation[(j * kDim) + k];
            }
            EXPECT_NEAR(dot, (i == j) ? 1.0F : 0.0F, 1e-3F)
                << "row " << i << " against " << j;
        }
    }
}

TEST(QuantizerTest, OpqDegradesToPlainPqWhenTheRotationCannotBeSolved) {
    // A training set spanning fewer directions than the dimension leaves the
    // Procrustes system rank-deficient. The trainer must fall back to the
    // identity rather than emit a non-orthonormal matrix.
    std::vector<float> degenerate(200 * kDim, 0.0F);
    for (std::size_t row = 0; row < 200; ++row) {
        degenerate[(row * kDim)] = static_cast<float>(row);
    }
    const auto opq = elips::quant::train(params_for(CodecId::opq),
                                         Metric::euclidean, kDim, degenerate, 200);

    std::vector<float> decoded(kDim);
    const auto code =
        opq->encode(std::span<const float>{degenerate.data(), kDim});
    opq->decode(code, decoded);
    for (const float v : decoded) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

// ------------------------------ codec identity -------------------------------
TEST(QuantizerTest, CodecIdentityRoundTripsThroughItsName) {
    for (const CodecId codec :
         {CodecId::none, CodecId::pq, CodecId::opq, CodecId::sq8}) {
        EXPECT_EQ(elips::quant::codec_from_string(elips::quant::to_string(codec)),
                  codec);
    }
    EXPECT_THROW((void)elips::quant::codec_from_string("lz4"),
                 elips::ConfigError);
    EXPECT_THROW((void)elips::quant::codec_from_string(nullptr),
                 elips::ConfigError);
}

TEST(QuantizerTest, ReportedCodecMatchesTheRequestedOne) {
    const auto rows = correlated_dataset(kRows, kDim);
    EXPECT_EQ(elips::quant::train(params_for(CodecId::pq), Metric::cosine, kDim,
                                  rows, kRows)
                  ->codec(),
              CodecId::pq);
    EXPECT_EQ(elips::quant::train(params_for(CodecId::opq), Metric::cosine, kDim,
                                  rows, kRows)
                  ->codec(),
              CodecId::opq);
    EXPECT_EQ(elips::quant::train(params_for(CodecId::sq8), Metric::cosine, kDim,
                                  rows, kRows)
                  ->codec(),
              CodecId::sq8);
}
