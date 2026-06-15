#include "elips/elips.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>

#include "elips/domain/Errors.hpp"
#include "elips/index_engine/IndexFactory.hpp"
#include "elips/kernel/LockManager.hpp"
#include "elips/storage/Serialization.hpp"
#include "elips/storage/WAL.hpp"
#include "elips/vector_engine/Metrics.hpp"

#ifdef ELIPS_GPU_ENABLED
#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/gpu_engine/GpuDeviceInfo.hpp"
#include "elips/gpu_engine/GpuMetricsSnapshot.hpp"
#endif

namespace elips {
namespace {

namespace fs = std::filesystem;

using detail::get;
using detail::get_payload;
using detail::get_string;
using detail::put;
using detail::put_payload;
using detail::put_string;

constexpr std::uint32_t snapshot_magic = 0xE1105E01U;
constexpr std::uint32_t snapshot_version = 1U;
constexpr std::uint32_t identity_magic = 0xE11D0001U;
constexpr const char* snapshot_file = "elips.snapshot";
constexpr const char* identity_file = "IDENTITY";
constexpr const char* lock_file = "LOCK";
constexpr const char* wal_file = "wal.log";

bool all_finite(std::span<const float> values) noexcept {
    for (const float v : values) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

}  // namespace

// --------------------------------- Vault ---------------------------------

Vault::Vault(std::string name, const Config& config)
    : name_(std::move(name)),
      config_(config),
      index_(make_index(config, config.dimension())) {}

Vector Vault::prepare(const Vector& vector) const {
    if (vector.dimension() != config_.dimension()) {
        throw DimensionMismatch{"vector dimension does not match vault"};
    }
    if (!all_finite(vector.values())) {
        throw InvalidVector{"vector contains NaN or Inf"};
    }
    return requires_normalization(config_.metric()) ? vector.normalized() : vector;
}

RecordID Vault::place(const Vector& vector, Payload payload,
                      std::optional<RecordID> id) {
    Vector prepared = prepare(vector);
    const RecordID record_id = id.value_or(RecordID::generate());
    if (wal_ != nullptr) {  // write-ahead before mutating in-memory state
        wal_->append_insert(name_, record_id, prepared.values(), payload);
    }
    index_->insert(record_id, prepared.values());
    records_[record_id] = Record{record_id, std::move(prepared), std::move(payload)};
    return record_id;
}

void Vault::place_many(const std::vector<Record>& records) {
    for (const auto& record : records) {
        const std::optional<RecordID> id =
            (record.id == RecordID{}) ? std::nullopt
                                      : std::optional<RecordID>{record.id};
        place(record.vector, record.payload, id);
    }
}

std::vector<SearchResult> Vault::seek(const Vector& query, std::size_t top,
                                      const Filter& filter,
                                      std::optional<float> threshold) const {
    const Vector prepared = prepare(query);
    const bool filtered = !filter.matches_all();

    // Choose how many index candidates to retrieve. With a filter we over-fetch
    // so post-filtering still yields up to `top`; a threshold (range) query
    // considers all candidates. (ef-expansion is a later optimizer refinement.)
    std::size_t fetch = top;
    if (threshold.has_value()) {
        fetch = records_.size();
    } else if (filtered) {
        fetch = std::min(records_.size(), top > 0 ? top * 20 : std::size_t{20});
    }
    fetch = std::max<std::size_t>(fetch, 1);

    const auto hits = index_->search(prepared.values(), fetch);
    std::vector<SearchResult> results;
    for (const auto& [id, dist] : hits) {
        if (threshold.has_value() && dist > *threshold) {
            continue;
        }
        const auto it = records_.find(id);
        const Payload data = (it != records_.end()) ? it->second.payload : Payload{};
        if (filtered && !filter.matches(data)) {
            continue;
        }
        results.push_back(SearchResult{id, dist, data});
        if (results.size() >= top) {
            break;
        }
    }
    return results;
}

std::vector<Record> Vault::scan(const Filter& filter, std::size_t offset,
                                std::size_t limit) const {
    std::vector<Record> out;
    std::size_t skipped = 0;
    for (const auto& [id, record] : records_) {
        if (!filter.matches(record.payload)) {
            continue;
        }
        if (skipped < offset) {
            ++skipped;
            continue;
        }
        if (out.size() >= limit) {
            break;
        }
        out.push_back(record);
    }
    return out;
}

std::optional<Record> Vault::fetch(const RecordID& id) const {
    const auto it = records_.find(id);
    if (it == records_.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool Vault::erase(const RecordID& id) {
    const auto it = records_.find(id);
    if (it == records_.end()) {
        return false;
    }
    if (wal_ != nullptr) {
        wal_->append_erase(name_, id);
    }
    index_->remove(id);
    records_.erase(it);
    return true;
}

VaultInfo Vault::info() const noexcept {
    return VaultInfo{records_.size(), config_.dimension(), config_.metric()};
}

// ----------------------------- ElipsInstance -----------------------------

ElipsInstance::ElipsInstance(std::string path, Config config, bool persistent,
                             std::optional<LockManager> lock)
    : path_(std::move(path)),
      config_(config),
      persistent_(persistent),
      lock_(std::move(lock)) {}

ElipsInstance::~ElipsInstance() {
    if (persistent_ && !closed_) {
        try {
            checkpoint();
        } catch (...) {
            // E.16: destructors must not throw. Best-effort checkpoint.
        }
    }
}

Vault& ElipsInstance::vault(const std::string& name) {
    const auto it = vaults_.find(name);
    if (it != vaults_.end()) {
        return *it->second;
    }
    auto created = std::make_unique<Vault>(name, config_);
    created->set_wal(wal_.get());
    Vault& ref = *created;
    vaults_.emplace(name, std::move(created));
    return ref;
}

Vault& ElipsInstance::adopt_vault(std::unique_ptr<Vault> vault) {
    Vault& ref = *vault;
    vaults_[vault->name()] = std::move(vault);
    return ref;
}

void ElipsInstance::attach_wal(std::unique_ptr<WAL> wal) {
    wal_ = std::move(wal);
    for (auto& [name, vault] : vaults_) {
        vault->set_wal(wal_.get());
    }
}

std::vector<std::string> ElipsInstance::list_vaults() const {
    std::vector<std::string> names;
    names.reserve(vaults_.size());
    for (const auto& [name, _] : vaults_) {
        names.push_back(name);
    }
    return names;
}

void ElipsInstance::checkpoint() {
    if (!persistent_) {
        return;
    }
    fs::create_directories(path_);
    const fs::path tmp = fs::path(path_) / (std::string(snapshot_file) + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw StorageError{"cannot open snapshot for writing"};
        }
        put<std::uint32_t>(out, snapshot_magic);
        put<std::uint32_t>(out, snapshot_version);
        put<std::uint16_t>(out, config_.dimension());
        put<std::uint8_t>(out, static_cast<std::uint8_t>(config_.metric()));
        put<std::uint32_t>(out, static_cast<std::uint32_t>(vaults_.size()));
        for (const auto& [name, vault] : vaults_) {
            put_string(out, name);
            const auto& records = vault->records();
            put<std::uint32_t>(out, static_cast<std::uint32_t>(records.size()));
            for (const auto& [id, record] : records) {
                out.write(reinterpret_cast<const char*>(id.bytes().data()),
                          static_cast<std::streamsize>(id.bytes().size()));
                const auto values = record.vector.values();
                put<std::uint16_t>(out, static_cast<std::uint16_t>(values.size()));
                out.write(reinterpret_cast<const char*>(values.data()),
                          static_cast<std::streamsize>(values.size_bytes()));
                put_payload(out, record.payload);
            }
        }
        if (!out) {
            throw StorageError{"error while writing snapshot"};
        }
    }
    // Atomic publish: rename temp over the live snapshot, then the WAL is
    // redundant (its records are captured) and may be truncated.
    fs::rename(tmp, fs::path(path_) / snapshot_file);
    if (wal_) {
        wal_->reset();
    }
}

void ElipsInstance::close() {
    if (closed_) {
        return;
    }
    checkpoint();
    for (auto& [name, vault] : vaults_) {
        vault->set_wal(nullptr);  // detach before releasing the WAL
    }
    wal_.reset();
    lock_.reset();  // release the advisory lock so the dir can be reopened
    closed_ = true;
}

// --------------------------------- open ----------------------------------

namespace {

struct Identity {
    std::uint16_t dimension{0};
    Metric metric{Metric::cosine};
    IndexType index{IndexType::graph};
};

void write_identity(const fs::path& file, const Config& config) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw StorageError{"cannot write IDENTITY"};
    }
    put<std::uint32_t>(out, identity_magic);
    put<std::uint32_t>(out, snapshot_version);
    put<std::uint16_t>(out, config.dimension());
    put<std::uint8_t>(out, static_cast<std::uint8_t>(config.metric()));
    put<std::uint8_t>(out, static_cast<std::uint8_t>(config.index()));
}

Identity read_identity(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in || get<std::uint32_t>(in) != identity_magic) {
        throw StorageError{"corrupt IDENTITY"};
    }
    (void)get<std::uint32_t>(in);  // version
    Identity id;
    id.dimension = get<std::uint16_t>(in);
    id.metric = static_cast<Metric>(get<std::uint8_t>(in));
    id.index = static_cast<IndexType>(get<std::uint8_t>(in));
    return id;
}

void load_snapshot(const fs::path& file, ElipsInstance& instance) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw StorageError{"cannot open snapshot for reading"};
    }
    if (get<std::uint32_t>(in) != snapshot_magic) {
        throw StorageError{"snapshot magic mismatch"};
    }
    if (get<std::uint32_t>(in) != snapshot_version) {
        throw StorageError{"unsupported snapshot version"};
    }
    (void)get<std::uint16_t>(in);  // dimension (authoritative copy is IDENTITY)
    (void)get<std::uint8_t>(in);   // metric

    const auto vault_count = get<std::uint32_t>(in);
    for (std::uint32_t v = 0; v < vault_count; ++v) {
        std::string name = get_string(in);
        auto vault = std::make_unique<Vault>(name, instance.config());
        const auto record_count = get<std::uint32_t>(in);
        for (std::uint32_t r = 0; r < record_count; ++r) {
            RecordID::Bytes bytes{};
            in.read(reinterpret_cast<char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
            const RecordID id{bytes};
            const auto dim = get<std::uint16_t>(in);
            std::vector<float> values(dim);
            in.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(dim) * sizeof(float));
            Payload payload = get_payload(in);
            if (!in) {
                throw StorageError{"truncated or corrupt snapshot"};
            }
            vault->place(Vector{std::move(values)}, std::move(payload), id);
        }
        instance.adopt_vault(std::move(vault));
    }
}

}  // namespace

std::unique_ptr<ElipsInstance> open(const std::string& path,
                                    const Config& config) {
    if (path == ":memory:") {
        if (config.dimension() == 0) {
            throw ConfigError{"in-memory database requires a dimension"};
        }
        return std::make_unique<ElipsInstance>(path, config, /*persistent=*/false);
    }

    fs::create_directories(path);
    LockManager lock{(fs::path(path) / lock_file).string()};  // single-writer

    const fs::path identity = fs::path(path) / identity_file;
    Config effective = config;
    if (fs::exists(identity)) {
        const Identity id = read_identity(identity);
        if (config.dimension() != 0 && config.dimension() != id.dimension) {
            throw ConfigError{"configured dimension conflicts with database"};
        }
        effective.dimension(id.dimension).metric(id.metric).index(id.index);
    } else {
        if (config.dimension() == 0) {
            throw ConfigError{"new database requires a dimension"};
        }
        write_identity(identity, config);
    }

    auto instance = std::make_unique<ElipsInstance>(path, effective,
                                                    /*persistent=*/true,
                                                    std::move(lock));

    const fs::path snapshot = fs::path(path) / snapshot_file;
    if (fs::exists(snapshot)) {
        load_snapshot(snapshot, *instance);
    }

    // Replay any writes that landed in the WAL after the last checkpoint.
    const fs::path walpath = fs::path(path) / wal_file;
    for (const auto& entry : WAL::replay(walpath)) {
        Vault& vault = instance->vault(entry.vault);
        if (entry.op == WAL::Op::insert) {
            vault.place(Vector{entry.vector}, entry.payload, entry.id);
        } else {
            vault.erase(entry.id);
        }
    }

    if (effective.durability() != Durability::ephemeral) {
        const bool sync = effective.durability() != Durability::relaxed;
        instance->attach_wal(std::make_unique<WAL>(walpath, sync));
    }

#ifdef ELIPS_GPU_ENABLED
    if (config.has_gpu() && config.gpu().policy != gpu::GpuPolicy::CpuOnly) {
        gpu::GpuDeviceManager manager;
        auto devices = manager.probe_all_devices();
        if (!devices.empty()) {
            instance->set_gpu_available(true);
            instance->set_gpu_info(devices.front());
        }
    }
#endif

    return instance;
}

// ------------------------------ Transaction ------------------------------

Transaction ElipsInstance::begin_transaction() { return Transaction{*this}; }

Transaction::~Transaction() {
    if (!done_) {
        rollback();  // discard buffered, never-applied ops
    }
}

void Transaction::enqueue_place(std::string vault, const Vector& vector,
                                Payload payload, std::optional<RecordID> id) {
    // Validate eagerly so commit() can never fail mid-batch (atomicity).
    if (vector.dimension() != db_->config().dimension()) {
        throw DimensionMismatch{"vector dimension does not match database"};
    }
    if (!all_finite(vector.values())) {
        throw InvalidVector{"vector contains NaN or Inf"};
    }
    ops_.push_back(PendingOp{false, std::move(vault), vector, std::move(payload),
                             std::move(id)});
}

void Transaction::enqueue_erase(std::string vault, const RecordID& id) {
    ops_.push_back(PendingOp{true, std::move(vault), Vector{}, Payload{}, id});
}

void Transaction::commit() {
    for (auto& op : ops_) {
        Vault& vault = db_->vault(op.vault);
        if (op.is_erase) {
            vault.erase(*op.id);
        } else {
            vault.place(op.vector, op.payload, op.id);
        }
    }
    ops_.clear();
    done_ = true;
}

RecordID TransactionVault::place(const Vector& vector, Payload payload,
                                 std::optional<RecordID> id) {
    const RecordID assigned = id.value_or(RecordID::generate());
    txn_->enqueue_place(vault_, vector, std::move(payload), assigned);
    return assigned;
}

void TransactionVault::erase(const RecordID& id) {
    txn_->enqueue_erase(vault_, id);
}

#ifdef ELIPS_GPU_ENABLED
gpu::GpuDeviceInfo ElipsInstance::gpu_info() const { return gpu_info_; }

gpu::GpuMetricsSnapshot ElipsInstance::gpu_stats() const { return gpu_stats_; }
#endif

}  // namespace elips
