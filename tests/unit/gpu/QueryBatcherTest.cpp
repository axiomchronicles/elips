#include <gtest/gtest.h>

#include <thread>
#include <chrono>

#include "elips/gpu_engine/QueryBatcher.hpp"

namespace elips::gpu {
namespace {

TEST(QueryBatcherTest, single_query_completes) {
    QueryBatcher batcher(100, 256);

    std::atomic<int> call_count{0};
    batcher.set_search_fn(
        [&](const std::vector<std::span<const float>>& queries, size_t k)
            -> std::vector<std::vector<elips::SearchResult>> {
            call_count++;
            std::vector<std::vector<elips::SearchResult>> results;
            for (size_t i = 0; i < queries.size(); ++i) {
                std::vector<elips::SearchResult> r;
                for (size_t j = 0; j < k; ++j) {
                    r.push_back({RecordID{}, static_cast<float>(j), {}});
                }
                results.push_back(std::move(r));
            }
            return results;
        });

    batcher.start();

    std::vector<float> query{1.0f, 2.0f, 3.0f};
    auto future = batcher.enqueue(query, 5);
    auto results = future.get();

    EXPECT_EQ(results.size(), 5u);
    EXPECT_GE(call_count.load(), 1);

    batcher.stop();
}

TEST(QueryBatcherTest, batch_coalesces_concurrent_queries) {
    QueryBatcher batcher(50000, 256);

    std::atomic<int> call_count{0};
    std::atomic<size_t> total_batched{0};
    batcher.set_search_fn(
        [&](const std::vector<std::span<const float>>& queries, size_t k)
            -> std::vector<std::vector<elips::SearchResult>> {
            call_count++;
            total_batched += queries.size();
            std::vector<std::vector<elips::SearchResult>> results;
            for (size_t i = 0; i < queries.size(); ++i) {
                results.push_back({});
            }
            return results;
        });

    batcher.start();

    std::vector<std::future<std::vector<elips::SearchResult>>> futures;
    for (int i = 0; i < 10; ++i) {
        std::vector<float> q{static_cast<float>(i), static_cast<float>(i * 2)};
        futures.push_back(batcher.enqueue(q, 1));
    }

    for (auto& f : futures) {
        f.get();
    }

    EXPECT_GE(total_batched.load(), 10u);
    EXPECT_LE(call_count.load(), 5);

    batcher.stop();
}

TEST(QueryBatcherTest, stats_are_accurate) {
    QueryBatcher batcher(100, 256);

    batcher.set_search_fn(
        [](const std::vector<std::span<const float>>& queries, size_t)
            -> std::vector<std::vector<elips::SearchResult>> {
            std::vector<std::vector<elips::SearchResult>> results;
            results.resize(queries.size());
            return results;
        });

    batcher.start();

    std::vector<float> q1{1.0f};
    auto f1 = batcher.enqueue(q1, 1);
    f1.get();

    auto stats = batcher.stats();
    EXPECT_GE(stats.queries_coalesced, 1u);
    EXPECT_GE(stats.kernel_launches, 1u);
    EXPECT_GE(stats.avg_batch_size, 0.0f);

    batcher.stop();
}

}  // namespace
}  // namespace elips::gpu