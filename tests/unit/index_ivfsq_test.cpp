#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/Errors.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/IndexFlat.hpp"
#include "elips/index_engine/IndexIVFScalarQuantizer.hpp"

namespace {

using elips::IndexFlat;
using elips::IndexIVFScalarQuantizer;
using elips::Metric;
using elips::RecordID;

constexpr std::uint16_t kDim = 32;
constexpr std::size_t kCount = 5000;
constexpr std::size_t kNList = 64;
constexpr std::size_t kK = 10;
constexpr std::size_t kQueries = 100;

std::vector<std::vector<float>> make_data(std::size_t n, std::uint16_t dim,
                                          unsigned seed) {
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
    }
    return data;
}

double recall_at(std::size_t nprobe) {
    const auto data = make_data(kCount, kDim, 7);
    const auto queries = make_data(kQueries, kDim, 99);

    IndexFlat oracle(Metric::euclidean, kDim);
    IndexIVFScalarQuantizer ivf(Metric::euclidean, kDim, kNList);

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

TEST(IndexIVFSQTest, RecallImprovesWithNprobeNearLossless) {
    const std::size_t probes[] = {1, 4, 8, 16, kNList};
    double prev = -1.0;
    double rfull = 0.0;
    for (std::size_t p : probes) {
        const double r = recall_at(p);
        std::fprintf(stderr, "[ivfsq] euclidean recall@10 nprobe%zu=%.4f\n", p, r);
        EXPECT_GE(r, prev - 0.02) << "nprobe " << p;
        prev = r;
        rfull = r;
    }
    // SQ8 is per-dimension near-lossless, so full-probe recall tracks the
    // IVFFlat oracle closely (floor from docs/_redesign/05_test_matrix.md).
    EXPECT_GE(rfull, 0.95);
}

TEST(IndexIVFSQTest, CompressesBelowFp32) {
    IndexIVFScalarQuantizer ivf(Metric::euclidean, kDim, kNList);
    // 32 code bytes vs 128 fp32 bytes -> 4x.
    EXPECT_LT(ivf.code_bytes(), static_cast<std::size_t>(kDim) * sizeof(float));
    EXPECT_EQ(ivf.code_bytes(), kDim);
}

TEST(IndexIVFSQTest, EmptyIndexReturnsNothing) {
    IndexIVFScalarQuantizer ivf(Metric::euclidean, 8, 8);
    std::vector<float> train(16 * 8);
    for (std::size_t i = 0; i < train.size(); ++i) {
        train[i] = static_cast<float>((i * 7) % 13);
    }
    ivf.train(train, 16);
    EXPECT_EQ(ivf.ntotal(), 0U);
    EXPECT_TRUE(ivf.search(std::vector<float>(8, 0.0F), 5).empty());
}

TEST(IndexIVFSQTest, KGreaterThanNtotalCaps) {
    IndexIVFScalarQuantizer ivf(Metric::euclidean, 8, 4);
    std::vector<float> train(16 * 8);
    for (std::size_t i = 0; i < train.size(); ++i) {
        train[i] = static_cast<float>((i * 3) % 11);
    }
    ivf.train(train, 16);
    ivf.nprobe = 4;
    ivf.add_with_ids(RecordID::generate(), std::vector<float>(8, 0.0F));
    ivf.add_with_ids(RecordID::generate(), std::vector<float>(8, 5.0F));
    EXPECT_EQ(ivf.search(std::vector<float>(8, 0.0F), 10).size(), 2U);
}

TEST(IndexIVFSQTest, RemoveDropsRecord) {
    IndexIVFScalarQuantizer ivf(Metric::euclidean, 8, 4);
    std::vector<float> train(16 * 8);
    for (std::size_t i = 0; i < train.size(); ++i) {
        train[i] = static_cast<float>((i * 5) % 9);
    }
    ivf.train(train, 16);
    ivf.nprobe = 4;
    const RecordID a = RecordID::generate();
    const RecordID b = RecordID::generate();
    ivf.add_with_ids(a, std::vector<float>(8, 0.0F));
    ivf.add_with_ids(b, std::vector<float>(8, 5.0F));
    ivf.remove(a);
    EXPECT_EQ(ivf.ntotal(), 1U);
    const auto hits = ivf.search(std::vector<float>(8, 5.0F), 5);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].first, b);
}

TEST(IndexIVFSQTest, ResetKeepsTrainingDropsData) {
    IndexIVFScalarQuantizer ivf(Metric::euclidean, 8, 4);
    std::vector<float> train(16 * 8);
    for (std::size_t i = 0; i < train.size(); ++i) {
        train[i] = static_cast<float>((i * 2) % 7);
    }
    ivf.train(train, 16);
    ivf.add_with_ids(RecordID::generate(), std::vector<float>(8, 0.0F));
    EXPECT_EQ(ivf.ntotal(), 1U);
    ivf.reset();
    EXPECT_EQ(ivf.ntotal(), 0U);
    EXPECT_TRUE(ivf.is_trained());
    ivf.add_with_ids(RecordID::generate(), std::vector<float>(8, 1.0F));
    EXPECT_EQ(ivf.ntotal(), 1U);
}

TEST(IndexIVFSQTest, AddBeforeTrainThrows) {
    IndexIVFScalarQuantizer ivf(Metric::euclidean, 8, 4);
    EXPECT_FALSE(ivf.is_trained());
    EXPECT_THROW(ivf.add_with_ids(RecordID::generate(),
                                  std::vector<float>(8, 0.0F)),
                 elips::ConfigError);
}

TEST(IndexIVFSQTest, TypeNameIsIvfsq) {
    IndexIVFScalarQuantizer ivf(Metric::euclidean, 8, 4);
    EXPECT_EQ(ivf.type_name(), "ivfsq");
}

}  // namespace
