// Recall and lifecycle tests for the quantized CPU indexes.
//
// The quantizer unit tests prove a codec reconstructs vectors; these prove the
// indexes still find the right neighbors once their storage is codes and every
// distance goes through asymmetric lookup. Recall is measured against an
// unquantized ExactIndex over the same data, which is the ground truth.
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "elips/index_engine/ExactIndex.hpp"
#include "elips/index_engine/HierarchicalGraphIndex.hpp"
#include "elips/quant_engine/Quantizer.hpp"

namespace {

using elips::GraphParams;
using elips::Metric;
using elips::RecordID;
using elips::quant::CodecId;
using elips::quant::QuantParams;

constexpr std::uint16_t kDim = 64;
constexpr std::size_t kCount = 4000;
constexpr std::size_t kQueries = 100;
constexpr std::size_t kTopK = 10;

std::vector<float> random_rows(std::size_t n, std::uint16_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> rows(n * dim);
    for (float& v : rows) {
        v = dist(rng);
    }
    return rows;
}

std::span<const float> row_at(const std::vector<float>& rows, std::size_t i) {
    return {rows.data() + (i * kDim), kDim};
}

elips::quant::QuantizerPtr make_quantizer(CodecId codec, Metric metric,
                                          const std::vector<float>& rows,
                                          std::size_t n) {
    QuantParams params;
    params.codec = codec;
    params.pq_dim = 16;
    params.pq_bits = 8;
    params.train_iters = 10;
    params.opq_iters = 2;
    return elips::quant::train(params, metric, kDim, rows, n);
}

// Fraction of the exact top-k that the index under test also returned.
double recall_at_k(const elips::IndexPort& index, const elips::ExactIndex& truth,
                   const std::vector<float>& queries, std::size_t query_count) {
    std::size_t hits = 0;
    std::size_t total = 0;
    for (std::size_t q = 0; q < query_count; ++q) {
        const auto query = row_at(queries, q);
        const auto expected = truth.search(query, kTopK);
        std::unordered_set<RecordID> expected_ids;
        for (const auto& [id, d] : expected) {
            expected_ids.insert(id);
        }
        for (const auto& [id, d] : index.search(query, kTopK)) {
            hits += expected_ids.count(id);
        }
        total += expected_ids.size();
    }
    return total == 0 ? 0.0
                      : static_cast<double>(hits) / static_cast<double>(total);
}

struct Fixture {
    std::vector<float> rows;
    std::vector<float> queries;
    std::vector<RecordID> ids;
    elips::ExactIndex truth;

    Fixture()
        : rows(random_rows(kCount, kDim, 21)),
          queries(random_rows(kQueries, kDim, 84)),
          ids(kCount),
          truth(Metric::euclidean, kDim) {
        for (std::size_t i = 0; i < kCount; ++i) {
            ids[i] = RecordID::generate();
            truth.insert(ids[i], row_at(rows, i));
        }
    }
};

}  // namespace

// ------------------------------- recall -------------------------------------
//
// The floors below are deliberately low, and that is a property of the test
// data rather than of the implementation. These fixtures are i.i.d. gaussian,
// which is the worst case for product quantization: there is no correlation
// between components for a codebook to exploit and no cluster structure to
// latch onto, so every subspace has to spend its 256 centroids covering an
// isotropic cloud. Real embeddings are strongly clustered and PQ does far
// better on them.
//
// Measured on this fixture (dim 64, recall@10 against the exact oracle):
//
//     pq_dim=8   8 bytes   32x   pq 0.30   opq 0.34
//     pq_dim=16  16 bytes  16x   pq 0.66   opq 0.61
//     pq_dim=32  32 bytes   8x   pq 0.85   opq 0.85
//     sq8        64 bytes   4x   sq 0.98
//
// The asserts encode two things worth protecting: the absolute floor at the
// default configuration, and the shape of the curve (more bytes buys more
// recall; SQ beats PQ at 4x the size). A regression that broke encoding or the
// ADC path would collapse both, well below these bounds.

TEST(QuantizedIndexTest, QuantizedExactIndexKeepsUsefulRecall) {
    const Fixture fixture;

    // An exhaustive scan over codes loses only what the codec's reconstruction
    // error costs -- no graph-navigation error on top.
    for (const auto [codec, floor] : {std::pair{CodecId::pq, 0.55},
                                      std::pair{CodecId::opq, 0.55},
                                      std::pair{CodecId::sq8, 0.95}}) {
        auto quantizer =
            make_quantizer(codec, Metric::euclidean, fixture.rows, kCount);
        elips::ExactIndex index(Metric::euclidean, kDim, quantizer);
        for (std::size_t i = 0; i < kCount; ++i) {
            index.insert(fixture.ids[i], row_at(fixture.rows, i));
        }

        const double recall =
            recall_at_k(index, fixture.truth, fixture.queries, kQueries);
        EXPECT_GE(recall, floor)
            << "codec " << elips::quant::to_string(codec) << " recall " << recall;
    }
}

TEST(QuantizedIndexTest, WiderCodesBuyBetterRecall) {
    // The tradeoff the feature exists to expose. If this ordering ever inverts,
    // the codebook is not being used to represent anything.
    const Fixture fixture;

    double previous = 0.0;
    for (const std::uint32_t pq_dim : {8U, 16U, 32U}) {
        QuantParams params;
        params.codec = CodecId::pq;
        params.pq_dim = pq_dim;
        params.pq_bits = 8;
        params.train_iters = 10;
        auto quantizer = elips::quant::train(params, Metric::euclidean, kDim,
                                             fixture.rows, kCount);
        ASSERT_EQ(quantizer->code_bytes(), pq_dim);

        elips::ExactIndex index(Metric::euclidean, kDim, quantizer);
        for (std::size_t i = 0; i < kCount; ++i) {
            index.insert(fixture.ids[i], row_at(fixture.rows, i));
        }
        const double recall =
            recall_at_k(index, fixture.truth, fixture.queries, kQueries);
        EXPECT_GT(recall, previous)
            << "pq_dim " << pq_dim << " did not improve on the narrower code";
        previous = recall;
    }

    // And the widest PQ code here is still narrower and less accurate than SQ,
    // which is why SQ is the right default when memory is a concern rather than
    // the binding constraint.
    auto sq = make_quantizer(CodecId::sq8, Metric::euclidean, fixture.rows, kCount);
    elips::ExactIndex sq_index(Metric::euclidean, kDim, sq);
    for (std::size_t i = 0; i < kCount; ++i) {
        sq_index.insert(fixture.ids[i], row_at(fixture.rows, i));
    }
    EXPECT_GT(recall_at_k(sq_index, fixture.truth, fixture.queries, kQueries),
              previous);
}

TEST(QuantizedIndexTest, QuantizedGraphIndexKeepsUsefulRecall) {
    const Fixture fixture;

    // The graph navigates using the same approximate distances it stores, so
    // this compounds codec error with beam-search error. The floors are
    // correspondingly below the exhaustive case above.
    for (const auto [codec, floor] : {std::pair{CodecId::pq, 0.45},
                                      std::pair{CodecId::opq, 0.45},
                                      std::pair{CodecId::sq8, 0.85}}) {
        auto quantizer =
            make_quantizer(codec, Metric::euclidean, fixture.rows, kCount);
        elips::HierarchicalGraphIndex index(Metric::euclidean, kDim,
                                            GraphParams{}, quantizer);
        for (std::size_t i = 0; i < kCount; ++i) {
            index.insert(fixture.ids[i], row_at(fixture.rows, i));
        }

        const double recall =
            recall_at_k(index, fixture.truth, fixture.queries, kQueries);
        EXPECT_GE(recall, floor)
            << "codec " << elips::quant::to_string(codec) << " recall " << recall;
    }
}

TEST(QuantizedIndexTest, CompressionRatioMatchesTheCodecWidth) {
    const Fixture fixture;
    const auto pq =
        make_quantizer(CodecId::pq, Metric::euclidean, fixture.rows, kCount);
    const auto sq =
        make_quantizer(CodecId::sq8, Metric::euclidean, fixture.rows, kCount);

    // What the feature is actually for: 16x on PQ at 16 subspaces, 4x on SQ,
    // per stored vector.
    EXPECT_EQ((kDim * sizeof(float)) / pq->code_bytes(), 16U);
    EXPECT_EQ((kDim * sizeof(float)) / sq->code_bytes(), 4U);
}

// ------------------------------ lifecycle -----------------------------------

TEST(QuantizedIndexTest, RemoveAndSizeTrackCorrectlyUnderCompression) {
    const Fixture fixture;
    auto quantizer =
        make_quantizer(CodecId::pq, Metric::euclidean, fixture.rows, kCount);
    elips::ExactIndex index(Metric::euclidean, kDim, quantizer);
    for (std::size_t i = 0; i < 100; ++i) {
        index.insert(fixture.ids[i], row_at(fixture.rows, i));
    }
    ASSERT_EQ(index.size(), 100U);

    // Removing from the middle must keep the remaining codes aligned with their
    // ids; a stride bug here silently returns the wrong record's distance.
    index.remove(fixture.ids[50]);
    EXPECT_EQ(index.size(), 99U);

    const auto hits = index.search(row_at(fixture.rows, 51), 1);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].first, fixture.ids[51]);
}

TEST(QuantizedIndexTest, GraphVacuumPreservesCodesWithoutRecompressing) {
    const Fixture fixture;
    auto quantizer =
        make_quantizer(CodecId::pq, Metric::euclidean, fixture.rows, kCount);

    GraphParams params;
    params.compaction_ratio = 0.0F;  // vacuum only when asked
    elips::HierarchicalGraphIndex index(Metric::euclidean, kDim, params,
                                        quantizer);
    for (std::size_t i = 0; i < 500; ++i) {
        index.insert(fixture.ids[i], row_at(fixture.rows, i));
    }

    const auto before = index.export_snapshot();
    ASSERT_TRUE(before.has_value());

    for (std::size_t i = 0; i < 100; ++i) {
        index.remove(fixture.ids[i]);
    }
    ASSERT_EQ(index.pending_removals(), 100U);
    index.vacuum();
    EXPECT_EQ(index.pending_removals(), 0U);
    EXPECT_EQ(index.size(), 400U);

    // The surviving rows must carry the *same* codes they had before the
    // rebuild. A vacuum that decoded and re-encoded would drift a generation
    // every compaction.
    const auto after = index.export_snapshot();
    ASSERT_TRUE(after.has_value());
    ASSERT_TRUE(after->pq.has_value());
    ASSERT_TRUE(before->pq.has_value());

    for (std::size_t i = 0; i < after->ids.size(); ++i) {
        const auto original = std::find(before->ids.begin(), before->ids.end(),
                                        after->ids[i]);
        ASSERT_NE(original, before->ids.end());
        const auto source =
            static_cast<std::size_t>(original - before->ids.begin());
        const std::size_t width = quantizer->code_bytes();
        for (std::size_t b = 0; b < width; ++b) {
            EXPECT_EQ(after->pq->codes[(i * width) + b],
                      before->pq->codes[(source * width) + b])
                << "row " << i << " byte " << b;
        }
    }
}

// ----------------------------- snapshots ------------------------------------

TEST(QuantizedIndexTest, SnapshotRoundTripPreservesRanking) {
    const Fixture fixture;
    auto quantizer =
        make_quantizer(CodecId::pq, Metric::euclidean, fixture.rows, kCount);

    elips::ExactIndex source(Metric::euclidean, kDim, quantizer);
    for (std::size_t i = 0; i < 500; ++i) {
        source.insert(fixture.ids[i], row_at(fixture.rows, i));
    }

    const auto snapshot = source.export_snapshot();
    ASSERT_TRUE(snapshot.has_value());
    ASSERT_TRUE(snapshot->pq.has_value());
    EXPECT_EQ(snapshot->pq->codec, CodecId::pq);
    // `vectors` stays fp32 so every backend can read it, even for a compressed
    // index.
    EXPECT_EQ(snapshot->vectors.size(), 500U * kDim);

    elips::ExactIndex restored(Metric::euclidean, kDim, quantizer);
    ASSERT_TRUE(restored.import_snapshot(*snapshot).has_value());
    EXPECT_EQ(restored.size(), source.size());

    for (std::size_t q = 0; q < 20; ++q) {
        const auto query = row_at(fixture.queries, q);
        const auto expected = source.search(query, kTopK);
        const auto actual = restored.search(query, kTopK);
        ASSERT_EQ(expected.size(), actual.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i].first, actual[i].first);
            EXPECT_FLOAT_EQ(expected[i].second, actual[i].second);
        }
    }
}

TEST(QuantizedIndexTest, GraphSnapshotRoundTripPreservesMembership) {
    const Fixture fixture;
    auto quantizer =
        make_quantizer(CodecId::sq8, Metric::cosine, fixture.rows, kCount);

    elips::HierarchicalGraphIndex source(Metric::cosine, kDim, GraphParams{},
                                         quantizer);
    for (std::size_t i = 0; i < 300; ++i) {
        source.insert(fixture.ids[i], row_at(fixture.rows, i));
    }

    const auto snapshot = source.export_snapshot();
    ASSERT_TRUE(snapshot.has_value());

    elips::HierarchicalGraphIndex restored(Metric::cosine, kDim, GraphParams{},
                                           quantizer);
    ASSERT_TRUE(restored.import_snapshot(*snapshot).has_value());
    EXPECT_EQ(restored.size(), 300U);

    // The graph is rebuilt with fresh random levels, so edge-for-edge identity
    // is not expected; membership and self-retrieval are.
    for (std::size_t i = 0; i < 50; ++i) {
        const auto hits = restored.search(row_at(fixture.rows, i), 1);
        ASSERT_FALSE(hits.empty());
        EXPECT_EQ(hits[0].first, fixture.ids[i]);
    }
}

// ---------------------------- unquantized parity -----------------------------

TEST(QuantizedIndexTest, NoQuantizerLeavesBehaviorAndTypeNameUnchanged) {
    // The default construction path must be untouched by all of the above.
    const Fixture fixture;
    elips::ExactIndex plain(Metric::euclidean, kDim);
    elips::HierarchicalGraphIndex graph(Metric::euclidean, kDim, GraphParams{});
    for (std::size_t i = 0; i < 200; ++i) {
        plain.insert(fixture.ids[i], row_at(fixture.rows, i));
        graph.insert(fixture.ids[i], row_at(fixture.rows, i));
    }

    EXPECT_EQ(plain.type_name(), "exact");
    EXPECT_EQ(graph.type_name(), "graph");

    // An unquantized index returns exact distances, so it matches the oracle.
    const auto hits = plain.search(row_at(fixture.rows, 7), 1);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].first, fixture.ids[7]);
    EXPECT_FLOAT_EQ(hits[0].second, 0.0F);
}

TEST(QuantizedIndexTest, QuantizedIndexesReportADistinctTypeName) {
    const Fixture fixture;
    auto quantizer =
        make_quantizer(CodecId::pq, Metric::euclidean, fixture.rows, kCount);
    const elips::ExactIndex exact(Metric::euclidean, kDim, quantizer);
    const elips::HierarchicalGraphIndex graph(Metric::euclidean, kDim,
                                              GraphParams{}, quantizer);
    EXPECT_EQ(exact.type_name(), "exact_quantized");
    EXPECT_EQ(graph.type_name(), "graph_quantized");
}
