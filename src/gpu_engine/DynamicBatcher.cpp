#include "elips/gpu_engine/DynamicBatcher.hpp"
#include <algorithm>

namespace elips::gpu {

DynamicBatcher::DynamicBatcher(size_t window_us, size_t max_batch)
    : window_us_(window_us), max_batch_(max_batch) {}

DynamicBatcher::~DynamicBatcher() { stop(); }

void DynamicBatcher::start() {
    running_ = true;
    worker_ = std::thread(&DynamicBatcher::worker_loop, this);
}

void DynamicBatcher::stop() {
    running_ = false;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

std::future<std::vector<elips::SearchResult>>
DynamicBatcher::enqueue(std::span<const float> query_vector, size_t k) {
    PendingQuery pq;
    pq.query.assign(query_vector.begin(), query_vector.end());
    pq.k = k;
    auto future = pq.promise.get_future();
    {
        const std::lock_guard lock(mutex_);
        pending_.push_back(std::move(pq));
    }
    cv_.notify_one();
    return future;
}

void DynamicBatcher::flush() {
    std::vector<PendingQuery> batch;
    {
        const std::lock_guard lock(mutex_);
        batch.swap(pending_);
    }
    if (batch.empty() || !search_fn_) return;

    total_launches_++;
    total_queries_ += batch.size();

    std::vector<std::span<const float>> queries;
    std::vector<size_t> ks;
    queries.reserve(batch.size());
    for (const auto& pq : batch) {
        queries.push_back(pq.query);
        ks.push_back(pq.k);
    }

    try {
        auto results = search_fn_(queries, ks.front());
        for (size_t i = 0; i < batch.size() && i < results.size(); ++i) {
            batch[i].promise.set_value(std::move(results[i]));
        }
    } catch (...) {
        for (auto& pq : batch) {
            try { pq.promise.set_exception(std::current_exception()); } catch (...) { (void)0; }
        }
    }
}

void DynamicBatcher::worker_loop() {
    while (running_) {
        std::unique_lock lock(mutex_);
        cv_.wait_for(lock, std::chrono::microseconds(window_us_),
                     [this] { return pending_.size() >= max_batch_ || !running_; });
        if (!running_ && pending_.empty()) break;
        if (pending_.empty()) continue;

        std::vector<PendingQuery> batch;
        const size_t take = std::min(pending_.size(), max_batch_);
        batch.assign(std::make_move_iterator(pending_.begin()),
                     std::make_move_iterator(pending_.begin() + static_cast<ptrdiff_t>(take)));
        pending_.erase(pending_.begin(), pending_.begin() + static_cast<ptrdiff_t>(take));
        lock.unlock();

        total_launches_++;
        total_queries_ += batch.size();

        if (search_fn_) {
            std::vector<std::span<const float>> queries;
            queries.reserve(batch.size());
            for (const auto& pq : batch) {
                queries.push_back(pq.query);
            }
            try {
                auto results = search_fn_(queries, batch[0].k);
                for (size_t i = 0; i < batch.size() && i < results.size(); ++i) {
                    batch[i].promise.set_value(std::move(results[i]));
                }
                for (size_t i = results.size(); i < batch.size(); ++i) {
                    batch[i].promise.set_value({});
                }
            } catch (...) {
                for (auto& pq : batch) {
                    try { pq.promise.set_exception(std::current_exception()); } catch (...) { (void)0; }
                }
            }
        }
    }
}

DynamicBatcher::BatchStats DynamicBatcher::stats() const noexcept {
    BatchStats s;
    s.kernel_launches = total_launches_.load();
    s.queries_coalesced = total_queries_.load();
    if (s.kernel_launches > 0) {
        s.avg_batch_size = static_cast<float>(s.queries_coalesced) /
                          static_cast<float>(s.kernel_launches);
    }
    return s;
}

void DynamicBatcher::set_search_fn(
    std::function<std::vector<std::vector<elips::SearchResult>>(
        const std::vector<std::span<const float>>&, size_t k)> fn) {
    search_fn_ = std::move(fn);
}

} // namespace elips::gpu