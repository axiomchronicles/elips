#ifndef ELIPS_GPU_ENGINE_QUERY_BATCHER_HPP
#define ELIPS_GPU_ENGINE_QUERY_BATCHER_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "elips/domain/SearchResult.hpp"

namespace elips::gpu {

class QueryBatcher {
public:
    QueryBatcher(size_t window_us = 500, size_t max_batch = 256);
    ~QueryBatcher();

    QueryBatcher(const QueryBatcher&) = delete;
    QueryBatcher& operator=(const QueryBatcher&) = delete;
    QueryBatcher(QueryBatcher&&) = delete;
    QueryBatcher& operator=(QueryBatcher&&) = delete;

    std::future<std::vector<elips::SearchResult>>
    enqueue(std::span<const float> query_vector, size_t k);

    void flush();

    struct BatchStats {
        size_t queries_coalesced{0};
        size_t kernel_launches{0};
        float avg_batch_size{0.0f};
        float p99_latency_us{0.0f};
    };

    BatchStats stats() const noexcept;

    void start();
    void stop();

    void set_search_fn(
        std::function<std::vector<std::vector<elips::SearchResult>>(
            const std::vector<std::span<const float>>&, size_t k)> fn);

private:
    struct PendingQuery {
        std::vector<float> query;
        size_t k;
        std::promise<std::vector<elips::SearchResult>> promise;
    };

    void worker_loop();

    size_t window_us_;
    size_t max_batch_;
    std::vector<PendingQuery> pending_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> running_{false};

    std::function<std::vector<std::vector<elips::SearchResult>>(
        const std::vector<std::span<const float>>&, size_t k)>
        search_fn_;

    std::atomic<size_t> total_queries_{0};
    std::atomic<size_t> total_launches_{0};
};

using DynamicBatcher [[deprecated("renamed to QueryBatcher")]] = QueryBatcher;

} // namespace elips::gpu

#endif