// M3 regression tests: HNSW tombstone compaction and adaptive beam width (F5),
// plus adaptive ef for post-filtered search.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "elips/elips.hpp"
#include "elips/index_engine/HierarchicalGraphIndex.hpp"

namespace {

std::vector<float> random_vector(std::mt19937& rng, std::size_t dim) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> v(dim);
    for (auto& value : v) {
        value = dist(rng);
    }
    return v;
}

TEST(M3IndexHealth, ChurnDoesNotGrowIndexUnbounded) {
    std::mt19937 rng(1234);
    constexpr std::uint16_t kDim = 16;
    elips::GraphParams params;
    params.compaction_ratio = 0.25F;
    elips::HierarchicalGraphIndex index(elips::Metric::euclidean, kDim, params);

    std::vector<elips::RecordID> live;
    for (int i = 0; i < 200; ++i) {
        const auto id = elips::RecordID::generate();
        index.insert(id, random_vector(rng, kDim));
        live.push_back(id);
    }
    ASSERT_EQ(index.size(), 200U);

    // Steady insert/delete churn: the live set stays ~200 but 4000 records pass
    // through. Without compaction the graph would hold all 4200 nodes forever.
    for (int round = 0; round < 20; ++round) {
        for (int i = 0; i < 200; ++i) {
            index.remove(live[static_cast<std::size_t>(i)]);
            const auto id = elips::RecordID::generate();
            index.insert(id, random_vector(rng, kDim));
            live[static_cast<std::size_t>(i)] = id;
        }
        // Tombstones never accumulate past the compaction threshold.
        EXPECT_LE(index.pending_removals(),
                  static_cast<std::size_t>(
                      params.compaction_ratio *
                      static_cast<float>(index.size() + index.pending_removals())) +
                      1U);
    }

    EXPECT_EQ(index.size(), 200U);
    // Graph nodes = live + tombstones; bounded, not 4200.
    EXPECT_LT(index.size() + index.pending_removals(), 400U);
}

TEST(M3IndexHealth, VacuumReclaimsTombstones) {
    std::mt19937 rng(99);
    constexpr std::uint16_t kDim = 8;
    elips::GraphParams params;
    params.compaction_ratio = 0.0F;  // disable auto compaction; test vacuum()
    elips::HierarchicalGraphIndex index(elips::Metric::euclidean, kDim, params);

    std::vector<elips::RecordID> ids;
    for (int i = 0; i < 100; ++i) {
        ids.push_back(elips::RecordID::generate());
        index.insert(ids.back(), random_vector(rng, kDim));
    }
    for (int i = 0; i < 60; ++i) {
        index.remove(ids[static_cast<std::size_t>(i)]);
    }
    EXPECT_EQ(index.pending_removals(), 60U);
    EXPECT_EQ(index.size(), 40U);

    index.vacuum();
    EXPECT_EQ(index.pending_removals(), 0U);
    EXPECT_EQ(index.size(), 40U);

    // Survivors remain findable after the rebuild.
    for (std::size_t i = 60; i < ids.size(); ++i) {
        const auto hits = index.search(random_vector(rng, kDim), 40);
        EXPECT_FALSE(hits.empty());
    }
}

TEST(M3IndexHealth, RecallHoldsAsDeleteRatioGrows) {
    std::mt19937 rng(7);
    constexpr std::uint16_t kDim = 12;
    constexpr std::size_t kTotal = 800;
    constexpr std::size_t kTop = 10;

    elips::GraphParams params;
    params.compaction_ratio = 0.0F;  // isolate the adaptive-ef behaviour
    elips::HierarchicalGraphIndex index(elips::Metric::euclidean, kDim, params);

    std::vector<elips::RecordID> ids;
    std::vector<std::vector<float>> vectors;
    for (std::size_t i = 0; i < kTotal; ++i) {
        ids.push_back(elips::RecordID::generate());
        vectors.push_back(random_vector(rng, kDim));
        index.insert(ids.back(), vectors.back());
    }

    const auto brute_force_top =
        [&](const std::vector<float>& query,
            const std::set<std::size_t>& removed) {
            std::vector<std::pair<float, std::size_t>> scored;
            for (std::size_t i = 0; i < kTotal; ++i) {
                if (removed.contains(i)) {
                    continue;
                }
                float sum = 0.0F;
                for (std::size_t d = 0; d < kDim; ++d) {
                    const float diff = query[d] - vectors[i][d];
                    sum += diff * diff;
                }
                scored.emplace_back(sum, i);
            }
            std::partial_sort(
                scored.begin(),
                scored.begin() + static_cast<std::ptrdiff_t>(
                                     std::min(kTop, scored.size())),
                scored.end());
            std::set<elips::RecordID> truth;
            for (std::size_t i = 0; i < std::min(kTop, scored.size()); ++i) {
                truth.insert(ids[scored[i].second]);
            }
            return truth;
        };

    std::set<std::size_t> removed;
    std::vector<std::vector<float>> queries;
    for (int q = 0; q < 20; ++q) {
        queries.push_back(random_vector(rng, kDim));
    }

    const auto measure = [&] {
        std::size_t returned = 0;
        std::size_t matched = 0;
        std::size_t expected = 0;
        for (const auto& query : queries) {
            const auto truth = brute_force_top(query, removed);
            const auto hits = index.search(query, kTop);
            returned += hits.size();
            expected += truth.size();
            for (const auto& [id, dist] : hits) {
                (void)dist;
                if (truth.contains(id)) {
                    ++matched;
                }
                // A deleted record must never be returned.
                EXPECT_FALSE(removed.contains(static_cast<std::size_t>(
                    std::find(ids.begin(), ids.end(), id) - ids.begin())));
            }
        }
        struct Result {
            double recall;
            double fill;
        };
        return Result{static_cast<double>(matched) /
                          static_cast<double>(std::max<std::size_t>(expected, 1)),
                      static_cast<double>(returned) /
                          static_cast<double>(std::max<std::size_t>(expected, 1))};
    };

    const auto baseline = measure();
    ASSERT_GT(baseline.recall, 0.7) << "baseline recall too low to draw a conclusion";
    EXPECT_GT(baseline.fill, 0.99);

    // Delete 50% of the corpus without compacting; tombstones now compete for
    // the search beam.
    for (std::size_t i = 0; i < kTotal; i += 2) {
        index.remove(ids[i]);
        removed.insert(i);
    }
    ASSERT_EQ(index.pending_removals(), kTotal / 2);

    const auto churned = measure();
    // The adaptive beam must still fill k live results...
    EXPECT_GT(churned.fill, 0.99)
        << "search returned fewer than k live hits at a 50% delete ratio";
    // ...and recall must not collapse relative to the pre-delete baseline.
    EXPECT_GT(churned.recall, baseline.recall - 0.15)
        << "recall degraded from " << baseline.recall << " to " << churned.recall;
}

// ---------------------- adaptive ef for filtered search -------------------

TEST(M3IndexHealth, FilteredSearchStillFillsTopK) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(8));
    auto& vault = db->vault("v");

    std::mt19937 rng(4242);
    // 1 in 50 records matches the filter, so a fixed top*20 over-fetch would
    // typically come back short.
    for (int i = 0; i < 2000; ++i) {
        elips::Payload payload;
        payload.emplace("bucket", static_cast<std::int64_t>(i % 50));
        vault.place(elips::Vector{random_vector(rng, 8)}, payload);
    }

    const auto filter = elips::Filter{}.field("bucket").equals(std::int64_t{7});
    const auto hits = vault.seek(elips::Vector{random_vector(rng, 8)}, 10, filter);
    EXPECT_EQ(hits.size(), 10U);
    for (const auto& hit : hits) {
        EXPECT_EQ(std::get<std::int64_t>(hit.data.at("bucket")), 7);
    }
}

TEST(M3IndexHealth, FilteredSearchReturnsAllMatchesWhenFewerThanTopK) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(4));
    auto& vault = db->vault("v");

    std::mt19937 rng(11);
    for (int i = 0; i < 500; ++i) {
        elips::Payload payload;
        payload.emplace("rare", static_cast<std::int64_t>(i < 3 ? 1 : 0));
        vault.place(elips::Vector{random_vector(rng, 4)}, payload);
    }

    const auto filter = elips::Filter{}.field("rare").equals(std::int64_t{1});
    const auto hits = vault.seek(elips::Vector{random_vector(rng, 4)}, 10, filter);
    // Only 3 records can match; the loop must terminate rather than spin.
    EXPECT_EQ(hits.size(), 3U);
}

// ------------------------- vault / database vacuum ------------------------

TEST(M3IndexHealth, DatabaseVacuumReclaimsAcrossVaults) {
    auto db = elips::open(":memory:", elips::Config{}.dimension(4).graph_params(
                                          elips::GraphParams{
                                              .max_connections = 16,
                                              .ef_construction = 200,
                                              .ef_search = 50,
                                              .compaction_ratio = 0.0F,
                                          }));
    std::mt19937 rng(5);
    for (const std::string name : {"a", "b"}) {
        auto& vault = db->vault(name);
        std::vector<elips::RecordID> ids;
        for (int i = 0; i < 50; ++i) {
            ids.push_back(vault.place(elips::Vector{random_vector(rng, 4)}));
        }
        for (int i = 0; i < 30; ++i) {
            vault.erase(ids[static_cast<std::size_t>(i)]);
        }
        EXPECT_EQ(vault.pending_removals(), 30U);
    }

    db->vacuum();
    EXPECT_EQ(db->vault("a").pending_removals(), 0U);
    EXPECT_EQ(db->vault("b").pending_removals(), 0U);
    EXPECT_EQ(db->vault("a").info().count, 20U);
    EXPECT_EQ(db->vault("b").info().count, 20U);
}

}  // namespace
