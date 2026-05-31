#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "Json.hpp"
#include "elips/elips.hpp"

namespace {

using elips::Config;

struct Args {
    std::string command;
    std::string path;
    std::map<std::string, std::string> options;
};

// Parse: elips <command> <db_path> [--key value]...
Args parse_args(int argc, char** argv) {
    Args args;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) == 0 && i + 1 < argc) {
            args.options[a.substr(2)] = argv[++i];
        } else {
            positional.push_back(a);
        }
    }
    if (!positional.empty()) args.command = positional[0];
    if (positional.size() > 1) args.path = positional[1];
    return args;
}

std::string option(const Args& a, const std::string& key,
                   const std::string& fallback = "") {
    const auto it = a.options.find(key);
    return it != a.options.end() ? it->second : fallback;
}

Config config_from_args(const Args& a) {
    Config config;
    config.dimension(
        static_cast<std::uint16_t>(std::stoul(option(a, "dimension", "0"))));
    const std::string metric = option(a, "metric");
    if (!metric.empty()) {
        config.metric(elips::metric_from_string(metric));
    }
    if (option(a, "index") == "exact") {
        config.index(elips::IndexType::exact);
    }
    return config;
}

// Open an existing database, or create one from CLI options (needs --dimension).
std::unique_ptr<elips::ElipsInstance> open_db(const Args& a) {
    namespace fs = std::filesystem;
    if (a.path != ":memory:" && fs::exists(fs::path(a.path) / "IDENTITY")) {
        return elips::open(a.path);
    }
    return elips::open(a.path, config_from_args(a));
}

void print_results(const std::vector<elips::SearchResult>& results) {
    for (const auto& r : results) {
        std::cout << R"({"id":")" << r.id.to_string() << R"(","distance":)"
                  << r.distance << R"(,"data":)";
        std::ostringstream data;
        bool first = true;
        data << '{';
        for (const auto& [k, v] : r.data) {
            if (!first) data << ',';
            first = false;
            data << '"' << elips::cli::json::escape(k)
                 << "\":" << elips::cli::json::dump_value(v);
        }
        data << '}';
        std::cout << data.str() << "}\n";
    }
}

int cmd_info(const Args& a) {
    auto db = open_db(a);
    const auto vaults = db->list_vaults();
    std::size_t total = 0;
    for (const auto& name : vaults) total += db->vault(name).info().count;
    std::cout << "path: " << a.path << "\n"
              << "dimension: " << db->config().dimension() << "\n"
              << "metric: " << elips::to_string(db->config().metric()) << "\n"
              << "index: "
              << (db->config().index() == elips::IndexType::graph ? "graph"
                                                                   : "exact")
              << "\n"
              << "vaults: " << vaults.size() << "\n"
              << "records: " << total << "\n";
    return 0;
}

int cmd_vaults(const Args& a) {
    auto db = open_db(a);
    for (const auto& name : db->list_vaults()) {
        std::cout << name << "\t" << db->vault(name).info().count << "\n";
    }
    return 0;
}

int cmd_query(const Args& a) {
    std::string eql = option(a, "eql");
    const std::string file = option(a, "file");
    if (!file.empty()) {
        std::ifstream in(file);
        std::stringstream ss;
        ss << in.rdbuf();
        eql = ss.str();
    }
    if (eql.empty()) {
        std::cerr << "query: provide --eql '<statement>' or --file <path>\n";
        return 2;
    }
    auto db = open_db(a);
    print_results(db->query(eql));
    return 0;
}

int cmd_checkpoint(const Args& a) {
    auto db = open_db(a);
    db->checkpoint();
    std::cout << "checkpoint complete\n";
    return 0;
}

int cmd_export(const Args& a) {
    const std::string vault = option(a, "vault");
    const std::string out_path = option(a, "output");
    auto db = open_db(a);
    std::ofstream out(out_path);
    std::size_t n = 0;
    for (const auto& record : db->vault(vault).scan()) {
        out << elips::cli::json::dump_record(record) << "\n";
        ++n;
    }
    std::cout << "exported " << n << " records to " << out_path << "\n";
    return 0;
}

int cmd_import(const Args& a) {
    const std::string vault_name = option(a, "vault");
    const std::string in_path = option(a, "input");
    auto db = open_db(a);
    auto& vault = db->vault(vault_name);
    std::ifstream in(in_path);
    std::string line;
    std::size_t n = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const elips::Record r = elips::cli::json::parse_record(line);
        vault.place(r.vector, r.payload, r.id);
        ++n;
    }
    db->checkpoint();
    std::cout << "imported " << n << " records into " << vault_name << "\n";
    return 0;
}

int cmd_verify(const Args& a) {
    try {
        auto db = open_db(a);  // replays WAL + validates snapshot
        std::size_t total = 0;
        for (const auto& name : db->list_vaults()) {
            total += db->vault(name).info().count;
        }
        std::cout << "OK: " << db->list_vaults().size() << " vaults, " << total
                  << " records\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CORRUPT: " << e.what() << "\n";
        return 1;
    }
}

int cmd_stats(const Args& a) {
    auto db = open_db(a);
    for (const auto& name : db->list_vaults()) {
        const auto info = db->vault(name).info();
        std::cout << "vault." << name << ".records: " << info.count << "\n";
    }
    return 0;
}

int cmd_bench(const Args& a) {
    const auto count = static_cast<std::size_t>(std::stoul(option(a, "count", "10000")));
    const auto dim = static_cast<std::uint16_t>(std::stoul(option(a, "dim", "128")));
    auto db = elips::open(a.path, Config{}.dimension(dim).metric(
                                      elips::Metric::cosine));
    auto& vault = db->vault("bench");

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    auto random_vec = [&] {
        std::vector<float> v(dim);
        for (auto& x : v) x = dist(rng);
        return elips::Vector{std::move(v)};
    };

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    for (std::size_t i = 0; i < count; ++i) vault.place(random_vec());
    const auto t1 = clock::now();

    constexpr int queries = 1000;
    for (int q = 0; q < queries; ++q) (void)vault.seek(random_vec(), 10);
    const auto t2 = clock::now();

    const double insert_s = std::chrono::duration<double>(t1 - t0).count();
    const double query_s = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "insert: " << count << " records in " << insert_s << "s ("
              << static_cast<std::size_t>(count / insert_s) << " rec/s)\n"
              << "search: " << queries << " queries, "
              << (query_s / queries) * 1e6 << " us/query avg\n";
    return 0;
}

void usage() {
    std::cerr << "usage: elips <command> <db_path> [options]\n"
              << "commands: info vaults query checkpoint export import verify "
                 "stats bench\n";
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    if (args.command.empty() ||
        (args.path.empty() && args.command != "help")) {
        usage();
        return 2;
    }
    try {
        if (args.command == "info") return cmd_info(args);
        if (args.command == "vaults") return cmd_vaults(args);
        if (args.command == "query") return cmd_query(args);
        if (args.command == "checkpoint") return cmd_checkpoint(args);
        if (args.command == "export") return cmd_export(args);
        if (args.command == "import") return cmd_import(args);
        if (args.command == "verify") return cmd_verify(args);
        if (args.command == "stats") return cmd_stats(args);
        if (args.command == "bench") return cmd_bench(args);
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
