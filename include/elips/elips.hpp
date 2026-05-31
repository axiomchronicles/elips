#ifndef ELIPS_ELIPS_HPP
#define ELIPS_ELIPS_HPP

#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/Record.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/domain/SearchResult.hpp"
#include "elips/domain/Vector.hpp"
#include "elips/index_engine/IndexPort.hpp"
#include "elips/kernel/LockManager.hpp"
#include "elips/metadata/Filter.hpp"

namespace elips {

class WAL;          // storage layer (kept out of the public API surface)
class Transaction;  // defined below

// Summary statistics for a vault.
struct VaultInfo {
    std::size_t count{0};
    std::uint16_t dimension{0};
    Metric metric{Metric::cosine};
};

// A named partition of records within a database. Owns its index and the
// authoritative record store used to hydrate search results.
class Vault {
public:
    Vault(std::string name, const Config& config);

    // Ingests one record: validates, normalizes (cosine), stores, indexes.
    // Returns the assigned id (UUIDv7 if none supplied).
    RecordID place(const Vector& vector, Payload payload = {},
                   std::optional<RecordID> id = std::nullopt);
    void place_many(const std::vector<Record>& records);

    // Top-k nearest neighbors, ascending by distance. An optional metadata
    // filter and distance threshold narrow the result (range search = large
    // top + threshold).
    [[nodiscard]] std::vector<SearchResult> seek(
        const Vector& query, std::size_t top, const Filter& filter = Filter{},
        std::optional<float> threshold = std::nullopt) const;

    // Iterate records matching a filter in insertion (UUIDv7) order.
    [[nodiscard]] std::vector<Record> scan(
        const Filter& filter = Filter{}, std::size_t offset = 0,
        std::size_t limit = std::numeric_limits<std::size_t>::max()) const;

    [[nodiscard]] std::optional<Record> fetch(const RecordID& id) const;
    bool erase(const RecordID& id);

    [[nodiscard]] VaultInfo info() const noexcept;
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    // Used by the persistence layer.
    [[nodiscard]] const std::map<RecordID, Record>& records() const noexcept {
        return records_;
    }
    void set_wal(WAL* wal) noexcept { wal_ = wal; }

private:
    [[nodiscard]] Vector prepare(const Vector& vector) const;

    std::string name_;
    Config config_;
    std::unique_ptr<IndexPort> index_;
    std::map<RecordID, Record> records_;  // ordered by UUIDv7 (≈ insertion time)
    WAL* wal_{nullptr};                    // non-owning; nullptr when in-memory
};

// Top-level database handle. One per directory. Owns all vaults and persistence.
// Checkpoints automatically on destruction for on-disk databases.
class ElipsInstance {
public:
    ElipsInstance(std::string path, Config config, bool persistent,
                  std::optional<LockManager> lock = std::nullopt);
    ~ElipsInstance();

    ElipsInstance(const ElipsInstance&) = delete;
    ElipsInstance& operator=(const ElipsInstance&) = delete;
    ElipsInstance(ElipsInstance&&) = delete;
    ElipsInstance& operator=(ElipsInstance&&) = delete;

    // Access or lazily create a vault.
    Vault& vault(const std::string& name);
    [[nodiscard]] std::vector<std::string> list_vaults() const;

    // Begin an atomic write transaction over this database.
    [[nodiscard]] Transaction begin_transaction();

    // Execute a single EQL statement. Query vectors referenced as $name in the
    // statement are supplied through `bindings`.
    [[nodiscard]] std::vector<SearchResult> query(
        const std::string& eql,
        const std::map<std::string, Vector>& bindings = {});

    // Flush all state to disk (no-op for in-memory databases).
    void checkpoint();
    void close();

    // Drop the handle without checkpointing, leaving only the WAL on disk.
    // Models an abrupt process exit; the next open() recovers via WAL replay.
    void abandon() noexcept { closed_ = true; }

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    // Used by the persistence layer to rehydrate a vault on open.
    Vault& adopt_vault(std::unique_ptr<Vault> vault);
    // Hands the live WAL to the instance and wires every vault to it.
    void attach_wal(std::unique_ptr<WAL> wal);

private:
    std::string path_;
    Config config_;
    bool persistent_;
    bool closed_{false};
    std::optional<LockManager> lock_;
    std::unique_ptr<WAL> wal_;
    std::map<std::string, std::unique_ptr<Vault>> vaults_;
};

// Open (or create) a database. Use ":memory:" for an ephemeral, non-persistent
// database. For a new on-disk database, config.dimension() must be > 0; on
// reopen the persisted dimension/metric are authoritative and a conflicting
// config throws ConfigError.
[[nodiscard]] std::unique_ptr<ElipsInstance> open(const std::string& path,
                                                  const Config& config = {});

class Transaction;

// Vault-scoped handle used inside a transaction; buffers writes into the txn.
class TransactionVault {
public:
    RecordID place(const Vector& vector, Payload payload = {},
                   std::optional<RecordID> id = std::nullopt);
    void erase(const RecordID& id);

private:
    friend class Transaction;
    TransactionVault(Transaction& txn, std::string vault)
        : txn_(&txn), vault_(std::move(vault)) {}
    Transaction* txn_;
    std::string vault_;
};

// Atomic, all-or-nothing batch of writes. Operations are buffered and applied
// only on commit(); an un-committed transaction is rolled back on destruction.
// Inputs are validated at buffer time so commit() cannot partially apply.
class Transaction {
public:
    explicit Transaction(ElipsInstance& db) : db_(&db) {}
    ~Transaction();

    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    Transaction(Transaction&&) = delete;
    Transaction& operator=(Transaction&&) = delete;

    [[nodiscard]] TransactionVault vault(const std::string& name) {
        return TransactionVault{*this, name};
    }

    void commit();
    void rollback() noexcept { ops_.clear(); done_ = true; }

private:
    friend class TransactionVault;
    struct PendingOp {
        bool is_erase{false};
        std::string vault;
        Vector vector;
        Payload payload;
        std::optional<RecordID> id;
    };
    void enqueue_place(std::string vault, const Vector& vector, Payload payload,
                       std::optional<RecordID> id);
    void enqueue_erase(std::string vault, const RecordID& id);

    ElipsInstance* db_;
    std::vector<PendingOp> ops_;
    bool done_{false};
};

}  // namespace elips

#endif  // ELIPS_ELIPS_HPP
