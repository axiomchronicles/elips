// ELIPS benchmark suite. Measures insert throughput, search latency
// percentiles, recall@10 vs the ExactIndex ground truth, and crash-recovery
// time. Self-contained (no dependency beyond the ELIPS library).
//
//   elips_bench [--count N] [--dim D] [--queries Q]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "elips/elips.hpp"
#include "elips/index_engine/ExactIndex.hpp"
#include "elips/index_engine/HierarchicalGraphIndex.hpp"

namespace {

using clock_type = std::chrono::steady_clock;
using elips::GraphParams;
using elips::Metric;
using elips::RecordID;

double seconds_since(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

std::vector<std::vector<float>> random_dataset(std::size_t count,
                                               std::uint16_t dim,
                                               std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<std::vector<float>> data(count, std::vector<float>(dim));
    for (auto& row : data) {
        for (auto& v : row) {
            v = dist(rng);
        }
    }
    return data;
}

void bench_insert_and_search(std::size_t count, std::uint16_t dim,
                             std::size_t queries) {
    std::mt19937 rng(7);
    const auto data = random_dataset(count, dim, rng);
    const auto probes = random_dataset(queries, dim, rng);

    elips::HierarchicalGraphIndex graph(Metric::euclidean, dim, GraphParams{});
    elips::ExactIndex exact(Metric::euclidean, dim);
    std::vector<RecordID> ids(count);

    const auto t_insert = clock_type::now();
    for (std::size_t i = 0; i < count; ++i) {
        ids[i] = RecordID::generate();
        graph.insert(ids[i], data[i]);
    }
    const double insert_s = seconds_since(t_insert);
    for (std::size_t i = 0; i < count; ++i) {
        exact.insert(ids[i], data[i]);
    }

    constexpr std::size_t k = 10;
    std::vector<double> latencies_us;
    latencies_us.reserve(queries);
    std::size_t recall_hits = 0;
    std::size_t recall_total = 0;
    for (const auto& probe : probes) {
        const auto t_q = clock_type::now();
        const auto approx = graph.search(probe, k);
        latencies_us.push_back(seconds_since(t_q) * 1e6);

        const auto truth = exact.search(probe, k);
        std::unordered_set<RecordID> truth_ids;
        for (const auto& [id, d] : truth) {
            truth_ids.insert(id);
        }
        for (const auto& [id, d] : approx) {
            if (truth_ids.count(id) != 0) {
                ++recall_hits;
            }
        }
        recall_total += truth.size();
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    auto pct = [&](double p) {
        return latencies_us[static_cast<std::size_t>(p * (latencies_us.size() - 1))];
    };
    const double recall =
        static_cast<double>(recall_hits) / static_cast<double>(recall_total);

    std::cout << "[insert]  " << count << " x " << dim << "  "
              << static_cast<std::size_t>(count / insert_s) << " rec/s ("
              << insert_s << "s)\n";
    std::cout << "[search]  p50=" << pct(0.50) << "us  p95=" << pct(0.95)
              << "us  p99=" << pct(0.99) << "us\n";
    std::cout << "[recall]  recall@" << k << " = " << recall << "\n";
}

void bench_recovery(std::size_t count, std::uint16_t dim) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "elips_bench_recovery";
    fs::remove_all(dir);

    std::mt19937 rng(11);
    const auto data = random_dataset(count, dim, rng);
    {
        auto db = elips::open(dir.string(),
                              elips::Config{}.dimension(dim).durability(
                                  elips::Durability::standard));
        auto& vault = db->vault("bench");
        for (const auto& row : data) {
            vault.place(elips::Vector{row});
        }
        db->abandon();  // crash before checkpoint -> recovery replays WAL
    }
    const auto t_recover = clock_type::now();
    auto db = elips::open(dir.string());
    const double recover_s = seconds_since(t_recover);
    std::cout << "[recovery] replayed " << db->vault("bench").info().count
              << " records in " << recover_s << "s\n";
    db->close();
    fs::remove_all(dir);
}

std::size_t arg_value(const std::map<std::string, std::string>& opts,
                      const std::string& key, std::size_t fallback) {
    const auto it = opts.find(key);
    return it != opts.end() ? std::stoul(it->second) : fallback;
}

}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> opts;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string key = argv[i];
        if (key.rfind("--", 0) == 0) {
            opts[key.substr(2)] = argv[i + 1];
        }
    }
    const auto count = arg_value(opts, "count", 20000);
    const auto dim = static_cast<std::uint16_t>(arg_value(opts, "dim", 128));
    const auto queries = arg_value(opts, "queries", 1000);

    std::cout << "ELIPS benchmark — count=" << count << " dim=" << dim
              << " queries=" << queries << "\n";
    bench_insert_and_search(count, dim, queries);
    bench_recovery(count / 4, dim);
    return 0;
}
