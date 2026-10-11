#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <span>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/Errors.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/IndexFlat.hpp"
#include "elips/index_engine/IndexIVFFlat.hpp"
#include "elips/index_engine/IndexIVFPQ.hpp"

namespace {

using elips::ElipsError;
using elips::Index;
using elips::IndexFlat;
using elips::IndexIVFFlat;
using elips::IndexIVFPQ;
using elips::Metric;
using elips::RecordID;

constexpr std::uint16_t kDim = 8;
constexpr std::size_t kCount = 200;
constexpr std::size_t kNList = 8;

std::vector<std::vector<float>> make_data(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<std::vector<float>> data(n, std::vector<float>(kDim));
    for (auto& v : data) {
        for (auto& x : v) {
            x = dist(rng);
        }
    }
    return data;
}

// Ground-truth: every id whose true-L2 distance to query is <= radius,
// ascending by distance. Computed by exhaustive IndexFlat search(k=ntotal).
std::vector<Index::Hit> brute_force_radius(
    const IndexFlat& oracle, std::span<const float> query, float radius,
    std::size_t ntotal) {
    auto all = oracle.search(query, ntotal);
    std::vector<Index::Hit> out;
    for (const auto& h : all) {
        if (h.second <= radius) {
            out.push_back(h);
        }
    }
    return out;
}

void expect_same(const std::vector<Index::Hit>& got,
                 const std::vector<Index::Hit>& want) {
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_EQ(got[i].first, want[i].first) << "id at rank " << i;
        EXPECT_FLOAT_EQ(got[i].second, want[i].second) << "dist at rank " << i;
        if (i > 0) {
            EXPECT_LE(got[i - 1].second, got[i].second) << "not ascending";
        }
    }
}

TEST(RangeSearch, FlatMatchesBruteForce) {
    const auto data = make_data(kCount, 1);
    IndexFlat idx(Metric::euclidean, kDim);
    std::vector<RecordID> ids;
    for (const auto& v : data) {
        ids.push_back(RecordID::generate());
        idx.add_with_ids(ids.back(), v);
    }

    const std::vector<float> query = data[0];
    for (float radius : {1.0F, 2.5F, 4.0F, 100.0F}) {
        const auto want = brute_force_radius(idx, query, radius, kCount);
        const auto got = idx.range_search(query, radius);
        expect_same(got, want);
    }
}

TEST(RangeSearch, IVFFlatExhaustiveMatchesBruteForce) {
    const auto data = make_data(kCount, 2);

    IndexFlat oracle(Metric::euclidean, kDim);
    IndexIVFFlat ivf(Metric::euclidean, kDim, kNList);

    std::vector<float> flat;
    for (const auto& v : data) {
        flat.insert(flat.end(), v.begin(), v.end());
    }
    ivf.train(flat, kCount);
    ivf.nprobe = kNList;  // exhaustive: must match the flat oracle exactly

    for (const auto& v : data) {
        const auto id = RecordID::generate();
        oracle.add_with_ids(id, v);
        ivf.add_with_ids(id, v);
    }

    const std::vector<float> query = data[0];
    for (float radius : {1.5F, 3.0F, 100.0F}) {
        const auto want = brute_force_radius(oracle, query, radius, kCount);
        const auto got = ivf.range_search(query, radius);
        expect_same(got, want);
    }
}

TEST(RangeSearch, BaseDefaultThrowsForUnsupportedIndex) {
    // IndexIVFPQ does not override range_search -> base default throws.
    const auto data = make_data(kCount, 3);
    IndexIVFPQ pq(Metric::euclidean, kDim, kNList, 4);
    std::vector<float> flat;
    for (const auto& v : data) {
        flat.insert(flat.end(), v.begin(), v.end());
    }
    pq.train(flat, kCount);
    pq.add_with_ids(RecordID::generate(), data[0]);

    EXPECT_THROW((void)pq.range_search(data[0], 1.0F), ElipsError);
}

}  // namespace
