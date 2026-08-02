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
    const std::string codec = option(a, "quantize");
    if (!codec.empty()) {
        elips::quant::QuantParams params;
        params.codec = elips::quant::codec_from_string(codec.c_str());
        const std::string pq_dim = option(a, "pq-dim");
        if (!pq_dim.empty()) {
            params.pq_dim = static_cast<std::uint32_t>(std::stoul(pq_dim));
        }
        const std::string pq_bits = option(a, "pq-bits");
        if (!pq_bits.empty()) {
            params.pq_bits = static_cast<std::uint32_t>(std::stoul(pq_bits));
        }
        config.quantization(params);
    }
    return config;
}

// Open an existing database, or create one from CLI options (needs --dimension).
//
// The config is passed either way. open() reconciles dimension, metric, and
// index against the persisted IDENTITY, so those flags stay advisory for an
// existing database, but options with no on-disk counterpart -- notably
// --quantize -- only reach the engine through here.
std::unique_ptr<elips::ElipsInstance> open_db(const Args& a) {
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
    for (const auto& name : vaults) {
        const auto info = db->vault(name).info();
        if (info.codec == elips::quant::CodecId::none) {
            continue;
        }
        const double ratio =
            (static_cast<double>(info.dimension) * sizeof(float)) /
            static_cast<double>(info.code_bytes);
        std::cout << "codec[" << name << "]: "
                  << elips::quant::to_string(info.codec) << " ("
                  << info.code_bytes << " B/vector, " << ratio << "x)\n";
    }
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
    const auto info = db->vault(vault).info();
    if (info.codec != elips::quant::CodecId::none) {
        // An export from a compressed vault is not a faithful backup: the
        // vectors are reconstructions, so a re-import will not reproduce the
        // original values.
        std::cerr << "warning: vault '" << vault << "' is compressed ("
                  << elips::quant::to_string(info.codec)
                  << "); exported vectors are reconstructions, not the original "
                     "values\n";
    }
    std::ofstream out(out_path);
    std::size_t n = 0;
    for (const auto& record : db->vault(vault).scan()) {
        out << elips::cli::json::dump_record(record) << "\n";
        ++n;
    }
    std::cout << "exported " << n << " records to " << out_path << "\n";
    return 0;
}

int cmd_quantize(const Args& a) {
    const std::string vault_name = option(a, "vault");
    auto db = open_db(a);

    const auto report = [&](const std::string& name) {
        const auto info = db->vault(name).info();
        const auto uncompressed =
            static_cast<double>(info.dimension) * sizeof(float);
        const double ratio =
            info.code_bytes == 0
                ? 1.0
                : uncompressed / static_cast<double>(info.code_bytes);
        std::cout << name << "\t" << elips::quant::to_string(info.codec) << "\t"
                  << info.code_bytes << " B/vector\t" << ratio << "x\t"
                  << info.count << " records\n";
    };

    if (vault_name.empty()) {
        db->quantize_all();
        for (const auto& name : db->list_vaults()) {
            report(name);
        }
    } else {
        db->quantize(vault_name);
        report(vault_name);
    }
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
                 "stats bench quantize\n"
              << "\nquantize [--vault NAME] [--quantize pq|opq|sq8] "
                 "[--pq-dim N] [--pq-bits N]\n"
              << "  Compresses stored vectors. Omit --vault for every vault.\n"
              << "  Trains a codebook from the resident data, so run it after\n"
              << "  ingest. Lossy: searches stay correct but return estimated\n"
              << "  distances, and exports become reconstructions.\n";
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
        if (args.command == "quantize") return cmd_quantize(args);
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
