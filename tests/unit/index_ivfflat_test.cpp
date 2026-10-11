#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/Errors.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/IndexFlat.hpp"
#include "elips/index_engine/IndexIVFFlat.hpp"

namespace {

using elips::IndexFlat;
using elips::IndexIVFFlat;
using elips::Metric;
using elips::RecordID;

constexpr std::uint16_t kDim = 32;
constexpr std::size_t kCount = 5000;
constexpr std::size_t kNList = 64;
constexpr std::size_t kK = 10;
constexpr std::size_t kQueries = 100;

// Deterministic dataset with mild cluster structure, the way real embeddings
// look (and the regime where an inverted file actually helps -- pure i.i.d.
// noise is adversarial for IVF because true neighbors straddle cell borders).
// cosine data is L2-normalized the way the vector engine normalizes on ingest.
std::vector<std::vector<float>> make_data(std::size_t n, std::uint16_t dim,
                                          Metric metric, unsigned seed) {
    constexpr std::size_t kClusters = 50;
    std::mt19937 rng(seed);
    std::normal_distribution<float> center_dist(0.0F, 1.0F);
    std::normal_distribution<float> jitter(0.0F, 0.15F);
    std::uniform_int_distribution<std::size_t> pick(0, kClusters - 1);

    std::vector<std::vector<float>> centers(kClusters, std::vector<float>(dim));
    for (auto& c : centers) {
        for (auto& x : c) {
            x = center_dist(rng);
        }
    }

    std::vector<std::vector<float>> data(n, std::vector<float>(dim));
    for (auto& v : data) {
        const auto& c = centers[pick(rng)];
        for (std::uint16_t d = 0; d < dim; ++d) {
            v[d] = c[d] + jitter(rng);
        }
        if (metric == Metric::cosine) {
            float norm = 0.0F;
            for (float x : v) {
                norm += x * x;
            }
            norm = std::sqrt(norm);
            if (norm > 0.0F) {
                for (float& x : v) {
                    x /= norm;
                }
            }
        }
    }
    return data;
}

double recall_at(Metric metric, std::size_t nprobe) {
    const auto data = make_data(kCount, kDim, metric, 7);
    const auto queries = make_data(kQueries, kDim, metric, 99);

    IndexFlat oracle(metric, kDim);
    IndexIVFFlat ivf(metric, kDim, kNList);

    // Train the coarse quantizer on the same vectors it will index.
    std::vector<float> flat;
    flat.reserve(kCount * kDim);
    for (const auto& v : data) {
        flat.insert(flat.end(), v.begin(), v.end());
    }
    ivf.train(flat, kCount);

    std::vector<RecordID> ids(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        ids[i] = RecordID::generate();
        oracle.add_with_ids(ids[i], data[i]);
        ivf.add_with_ids(ids[i], data[i]);
    }
    EXPECT_EQ(ivf.ntotal(), kCount);

    ivf.nprobe = nprobe;

    std::size_t hits = 0;
    std::size_t total = 0;
    for (const auto& q : queries) {
        const auto truth = oracle.search(q, kK);
        const auto approx = ivf.search(q, kK);
        std::unordered_set<RecordID> truth_ids;
        for (const auto& [id, d] : truth) {
            truth_ids.insert(id);
        }
        for (const auto& [id, d] : approx) {
            if (truth_ids.count(id) != 0) {
                ++hits;
            }
        }
        total += truth.size();
    }
    return static_cast<double>(hits) / static_cast<double>(total);
}

TEST(IndexIVFFlatTest, RecallMeetsThresholdAllMetrics) {
    for (const Metric metric :
         {Metric::euclidean, Metric::cosine, Metric::dot_product}) {
        const double r8 = recall_at(metric, 8);
        const double r16 = recall_at(metric, 16);
        const double rfull = recall_at(metric, kNList);
        std::fprintf(stderr,
                     "[ivfflat] metric=%d recall@10 nprobe8=%.4f nprobe16=%.4f "
                     "nprobe%zu=%.4f\n",
                     static_cast<int>(metric), r8, r16, kNList, rfull);
        EXPECT_GE(r16, 0.90) << "metric " << static_cast<int>(metric);
        // Scanning every list is exhaustive, so IVFFlat matches the oracle.
        EXPECT_GE(rfull, 0.999) << "metric " << static_cast<int>(metric);
    }
}

TEST(IndexIVFFlatTest, EmptyIndexReturnsNothing) {
    IndexIVFFlat ivf(Metric::euclidean, 4, 8);
    std::vector<float> train(64, 0.5F);  // 16 rows of dim 4
    for (std::size_t i = 0; i < 16; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            train[i * 4 + j] = static_cast<float>((i + j) % 5);
        }
    }
    ivf.train(train, 16);
    EXPECT_EQ(ivf.ntotal(), 0U);
    EXPECT_TRUE(ivf.search(std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F}, 5).empty());
}

TEST(IndexIVFFlatTest, KGreaterThanNtotalCaps) {
    IndexIVFFlat ivf(Metric::euclidean, 2, 4);
    std::vector<float> train{0, 0, 1, 1, 2, 2, 3, 3};  // 4 rows
    ivf.train(train, 4);
    ivf.nprobe = 4;
    ivf.add_with_ids(RecordID::generate(), std::vector<float>{0.0F, 0.0F});
    ivf.add_with_ids(RecordID::generate(), std::vector<float>{1.0F, 1.0F});
    EXPECT_EQ(ivf.search(std::vector<float>{0.0F, 0.0F}, 10).size(), 2U);
}

TEST(IndexIVFFlatTest, RemoveDropsRecord) {
    IndexIVFFlat ivf(Metric::euclidean, 2, 4);
    std::vector<float> train{0, 0, 1, 1, 2, 2, 3, 3};
    ivf.train(train, 4);
    ivf.nprobe = 4;
    const RecordID a = RecordID::generate();
    const RecordID b = RecordID::generate();
    ivf.add_with_ids(a, std::vector<float>{0.0F, 0.0F});
    ivf.add_with_ids(b, std::vector<float>{3.0F, 3.0F});
    ivf.remove(a);
    EXPECT_EQ(ivf.ntotal(), 1U);
    const auto hits = ivf.search(std::vector<float>{0.0F, 0.0F}, 5);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].first, b);
}

TEST(IndexIVFFlatTest, ResetKeepsTrainingDropsData) {
    IndexIVFFlat ivf(Metric::euclidean, 2, 4);
    std::vector<float> train{0, 0, 1, 1, 2, 2, 3, 3};
    ivf.train(train, 4);
    ivf.add_with_ids(RecordID::generate(), std::vector<float>{0.0F, 0.0F});
    EXPECT_EQ(ivf.ntotal(), 1U);
    ivf.reset();
    EXPECT_EQ(ivf.ntotal(), 0U);
    EXPECT_TRUE(ivf.is_trained());
    // Still usable without retraining.
    ivf.add_with_ids(RecordID::generate(), std::vector<float>{1.0F, 1.0F});
    EXPECT_EQ(ivf.ntotal(), 1U);
}

TEST(IndexIVFFlatTest, AddBeforeTrainThrows) {
    IndexIVFFlat ivf(Metric::euclidean, 2, 4);
    EXPECT_FALSE(ivf.is_trained());
    EXPECT_THROW(ivf.add_with_ids(RecordID::generate(),
                                  std::vector<float>{0.0F, 0.0F}),
                 elips::ConfigError);
}

TEST(IndexIVFFlatTest, TypeNameIsIvfflat) {
    IndexIVFFlat ivf(Metric::cosine, 4, 8);
    EXPECT_EQ(ivf.type_name(), "ivfflat");
}

}  // namespace
