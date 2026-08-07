#ifndef ELIPS_ELIPS_HPP
#define ELIPS_ELIPS_HPP

#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
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
#include "elips/metadata/MetadataIndex.hpp"

#ifdef ELIPS_GPU_ENABLED
#include "elips/gpu_engine/GpuDeviceInfo.hpp"
#include "elips/gpu_engine/GpuMetricsSnapshot.hpp"
#include "elips/gpu_engine/GpuPort.hpp"
#endif

#if defined(_WIN32) && defined(__MINGW32__)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace elips {

#if defined(_WIN32) && defined(__MINGW32__)
class SharedMutex {
public:
    SharedMutex() noexcept {
        InitializeSRWLock(&srw_);
    }
    ~SharedMutex() noexcept = default;

    SharedMutex(const SharedMutex&) = delete;
    SharedMutex& operator=(const SharedMutex&) = delete;
    SharedMutex(SharedMutex&&) = delete;
    SharedMutex& operator=(SharedMutex&&) = delete;

    void lock() noexcept {
        AcquireSRWLockExclusive(&srw_);
    }

    bool try_lock() noexcept {
        return TryAcquireSRWLockExclusive(&srw_) != 0;
    }

    void unlock() noexcept {
        ReleaseSRWLockExclusive(&srw_);
    }

    void lock_shared() noexcept {
        AcquireSRWLockShared(&srw_);
    }

    bool try_lock_shared() noexcept {
        return TryAcquireSRWLockShared(&srw_) != 0;
    }

    void unlock_shared() noexcept {
        ReleaseSRWLockShared(&srw_);
    }

private:
    SRWLOCK srw_{SRWLOCK_INIT};
};
#else
using SharedMutex = std::shared_mutex;
#endif

class WAL;
class Transaction;

// Summary statistics for a vault.
struct VaultInfo {
    std::size_t count{0};
    std::uint16_t dimension{0};
    Metric metric{Metric::cosine};
    // Compression state. `codec` is none until quantize() has trained a
    // codebook, even when the config selects one; `code_bytes` is the width of
    // one stored code, against dimension * 4 uncompressed.
    quant::CodecId codec{quant::CodecId::none};
    std::size_t code_bytes{0};
};

enum class QueryStrategy {
    ann_index,
    exact_candidates,
    full_scan,
    text_probe,
    hybrid_fusion,
};

struct QueryPlan {
    QueryStrategy strategy{QueryStrategy::ann_index};
    std::size_t candidate_count{0};
    bool metadata_accelerated{false};
    bool gpu_index{false};
    std::string index_type;
    quant::CodecId codec{quant::CodecId::none};
};

// A named partition of records within a database. Owns its index and the
// authoritative record store used to hydrate search results.
//
// Thread safety: every public method is safe to call concurrently from multiple
// threads on the same object. Readers share access; mutators are exclusive.
// A cross-process `flock` (see LockManager) keeps other *processes* out; this
// mutex is what keeps threads inside this process from racing.
class Vault {
public:
    Vault(std::string name, const Config& config
#ifdef ELIPS_GPU_ENABLED
          , gpu::GpuPort* gpu_backend = nullptr
#endif
    );

    RecordID place(const Vector& vector, Payload payload = {},
                   std::optional<RecordID> id = std::nullopt,
                   std::optional<DocumentAttachment> document = std::nullopt,
                   std::optional<ChunkInfo> chunk = std::nullopt,
                   std::optional<EmbeddingLineage> lineage = std::nullopt);
    RecordID place_document(
        std::string text, Payload payload = {},
        std::optional<RecordID> id = std::nullopt,
        std::optional<ChunkInfo> chunk = std::nullopt,
        std::optional<EmbeddingLineage> lineage = std::nullopt);
    void place_many(const std::vector<Record>& records);

    // Stores a record as-is, bypassing normalization and encoding when it
    // already carries a code. This is the load path: snapshot, segment, and WAL
    // replay all use it so a persisted code is restored verbatim rather than
    // decoded and re-encoded. Going through place() instead would run one
    // generation of loss per reopen, unbounded across restarts.
    void place_record(const Record& record);

    [[nodiscard]] std::vector<SearchResult> seek(
        const Vector& query, std::size_t top, const Filter& filter = Filter{},
        std::optional<float> threshold = std::nullopt) const;
    [[nodiscard]] std::vector<SearchResult> seek_text(
        std::string_view text, std::size_t top, const Filter& filter = Filter{},
        std::optional<float> threshold = std::nullopt) const;
    [[nodiscard]] std::vector<SearchResult> seek_hybrid(
        const Vector& query, std::string_view text, std::size_t top,
        const Filter& filter = Filter{},
        std::optional<float> threshold = std::nullopt,
        float lexical_weight = 0.25F) const;
    [[nodiscard]] QueryPlan explain_seek(
        const Vector& query, std::size_t top, const Filter& filter = Filter{},
        std::optional<float> threshold = std::nullopt,
        bool has_text_component = false) const;

    [[nodiscard]] std::vector<Record> scan(
        const Filter& filter = Filter{}, std::size_t offset = 0,
        std::size_t limit = std::numeric_limits<std::size_t>::max()) const;

    [[nodiscard]] std::optional<Record> fetch(const RecordID& id) const;
    bool erase(const RecordID& id);

    [[nodiscard]] VaultInfo info() const noexcept;
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    // Snapshot of the record store. Returns a copy because the live map is
    // guarded by the vault mutex and a reference could be mutated underneath a
    // concurrent reader.
    [[nodiscard]] std::map<RecordID, Record> records() const;

    void set_wal(WAL* wal) noexcept { wal_ = wal; }
    void set_read_only(bool read_only) noexcept;
    [[nodiscard]] bool read_only() const noexcept;
    void rebuild_index();

    // Reclaim index space held by deleted records. Cheap no-op when there is
    // nothing tombstoned; the graph index also self-compacts once tombstones
    // pass GraphParams::compaction_ratio.
    void vacuum();
    // Records deleted from the index but not yet reclaimed.
    [[nodiscard]] std::size_t pending_removals() const noexcept;

    // Trains a codebook over the resident vectors and compresses the vault in
    // place, freeing the fp32 copies. Requires a codec to have been selected in
    // the config; throws ConfigError otherwise, or if the vault is empty or has
    // already been quantized.
    //
    // Product quantization cannot encode the first record -- a codebook has to
    // be learned from real data first -- so compression is an explicit
    // transition rather than something the config alone turns on. Before it, a
    // vault ingests and serves full fp32; after it, inserts encode on arrival.
    // Queries work correctly in both states.
    //
    // This is a maintenance operation in the same class as compact(): it holds
    // the vault's writer lock for its whole duration, so concurrent writes block
    // (reads too, briefly). Training samples at most quant::train_sample_cap
    // rows, so its cost tracks the dimension and codebook size rather than the
    // vault size, but encoding is still linear in the record count.
    void quantize();
    [[nodiscard]] bool quantized() const noexcept;
    [[nodiscard]] quant::CodecId codec() const noexcept;

    // Effective configuration, including any attached codebook. Checkpointing
    // reads the quantizer through this to persist it.
    [[nodiscard]] const Config& config() const noexcept { return config_; }
    // Installs a codebook read back from disk. Called during load, before any
    // record is placed, so coded records can be interpreted as they arrive.
    void install_quantizer(quant::QuantizerPtr quantizer);

    // Refuse further mutations. Called when the owning database closes: writes
    // after close would mutate memory that is never checkpointed and never
    // WAL-logged, i.e. silently vanish.
    void seal() noexcept;
    [[nodiscard]] bool sealed() const noexcept;

    // Transaction rollback hook: restores one record to a prior state (or
    // removes it when `previous` is empty) touching only in-memory structures.
    // Deliberately skips the WAL and the read-only guard -- the WAL's
    // transaction markers already exclude the uncommitted batch from replay,
    // and undo must work even when the WAL write is what failed.
    void restore_for_undo(const RecordID& id,
                          const std::optional<Record>& previous);

private:
    [[nodiscard]] QueryPlan plan_seek(
        const Vector& prepared, std::size_t top, const Filter& filter,
        std::optional<float> threshold, bool has_text_component) const;
    [[nodiscard]] Vector prepare(const Vector& vector) const;
    [[nodiscard]] std::vector<SearchResult> search_records(
        const Vector& prepared, std::size_t top, const Filter& filter,
        std::optional<float> threshold,
        const std::vector<const Record*>* subset = nullptr) const;
    void ensure_writable() const;

    // `_locked` suffix: caller already holds mutex_. Public entry points take
    // the lock and delegate here, so nested calls cannot self-deadlock
    // (std::shared_mutex is not recursive).
    RecordID place_locked(const Vector& vector, Payload payload,
                          std::optional<RecordID> id,
                          std::optional<DocumentAttachment> document,
                          std::optional<ChunkInfo> chunk,
                          std::optional<EmbeddingLineage> lineage);
    bool erase_locked(const RecordID& id);
    [[nodiscard]] std::vector<SearchResult> seek_locked(
        const Vector& query, std::size_t top, const Filter& filter,
        std::optional<float> threshold) const;
    void rebuild_index_locked();
    void place_record_locked(const Record& record);

    // Full-precision view of a stored record: the vector itself when
    // uncompressed, a decode of its code otherwise.
    [[nodiscard]] Vector reconstruct(const Record& record) const;
    // Materializes a record for a caller: reconstructs the vector and stamps
    // the codec so an approximate value is labelled as one.
    [[nodiscard]] Record hydrate(const Record& record) const;
    [[nodiscard]] const quant::QuantizerPtr& quantizer() const noexcept {
        return config_.quantizer();
    }

    std::string name_;
    Config config_;
    std::unique_ptr<IndexPort> index_;
    std::map<RecordID, Record> records_;
    MetadataIndex metadata_index_;
    WAL* wal_{nullptr};
    bool read_only_{false};
    bool sealed_{false};
#ifdef ELIPS_GPU_ENABLED
    gpu::GpuPort* gpu_backend_{nullptr};
#endif
    // Guards every member above. Mutable so const (read) methods can share-lock.
    mutable SharedMutex mutex_;
};

// Top-level database handle. One per directory. Owns all vaults and persistence.
// Checkpoints automatically on destruction for on-disk databases.
//
// Thread safety: safe to share across threads. The instance mutex guards the
// vault registry and lifecycle state; each Vault carries its own reader/writer
// lock, so concurrent searches on the same vault do not serialize.
class ElipsInstance {
public:
    ElipsInstance(std::string path, Config config, bool persistent,
                  std::optional<LockManager> lock = std::nullopt);
    ~ElipsInstance();

    ElipsInstance(const ElipsInstance&) = delete;
    ElipsInstance& operator=(const ElipsInstance&) = delete;
    ElipsInstance(ElipsInstance&&) = delete;
    ElipsInstance& operator=(ElipsInstance&&) = delete;

    Vault& vault(const std::string& name);
    [[nodiscard]] std::vector<std::string> list_vaults() const;

    [[nodiscard]] Transaction begin_transaction();

    [[nodiscard]] std::vector<SearchResult> query(
        const std::string& eql,
        const std::map<std::string, Vector>& bindings = {});

    void checkpoint();
    void compact();

    // Compresses a vault (or every vault) in place. See Vault::quantize().
    // For a persistent database this checkpoints afterwards, publishing the
    // codebook and the encoded records durably before the WAL is truncated.
    void quantize(const std::string& vault_name);
    void quantize_all();
    // Reclaim tombstoned index space across every vault. Unlike compact(), this
    // does not rewrite the on-disk snapshot and works on in-memory databases.
    void vacuum();
    void close();
    void abandon() noexcept;
    [[nodiscard]] WAL* wal() const noexcept { return wal_.get(); }

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    // Directory this database was opened from, or ":memory:".
    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] bool persistent() const noexcept { return persistent_; }
    [[nodiscard]] bool closed() const noexcept { return closed_; }

#ifdef ELIPS_GPU_ENABLED
    [[nodiscard]] gpu::GpuDeviceInfo gpu_info() const;
    [[nodiscard]] gpu::GpuMetricsSnapshot gpu_stats() const;
    void set_gpu_available(bool available) noexcept { gpu_available_ = available; }
    void set_gpu_info(gpu::GpuDeviceInfo info) noexcept { gpu_info_ = info; }
    void set_gpu_backend(std::unique_ptr<gpu::GpuPort> backend) noexcept {
        gpu_backend_ = std::move(backend);
    }
#endif

    Vault& adopt_vault(std::unique_ptr<Vault> vault);
    void attach_wal(std::unique_ptr<WAL> wal);

private:
    friend class Transaction;

    Vault& vault_locked(const std::string& name);
    void checkpoint_locked();

    std::string path_;
    Config config_;
    bool persistent_;
    bool closed_{false};
    std::optional<LockManager> lock_;
    std::unique_ptr<WAL> wal_;
#ifdef ELIPS_GPU_ENABLED
    gpu::GpuDeviceInfo gpu_info_;
    gpu::GpuMetricsSnapshot gpu_stats_;
    bool gpu_available_{false};
    // Vaults and GPU indexes only observe the backend, so the owner must
    // outlive every vault teardown path.
    std::unique_ptr<gpu::GpuPort> gpu_backend_;
#endif
    std::map<std::string, std::unique_ptr<Vault>> vaults_;
    // Guards the vault registry, the WAL handle, and lifecycle flags. Vault
    // contents have their own finer-grained lock.
    mutable std::recursive_mutex mutex_;
};

[[nodiscard]] std::unique_ptr<ElipsInstance> open(const std::string& path,
                                                  const Config& config = {});

class Transaction;

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

    // Applies every enqueued op or none: a failure partway through (read-only
    // vault, disk-full WAL write) undoes the ops already applied and rethrows.
    void commit();
    // Discards the batch. Safe to call after a failed commit(); the failed
    // commit has already restored the pre-batch state.
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
    // What the vault held before an op was applied, so it can be restored.
    struct UndoEntry {
        std::string vault;
        RecordID id;
        std::optional<Record> previous;  // nullopt = record did not exist
    };
    void enqueue_place(std::string vault, const Vector& vector, Payload payload,
                       std::optional<RecordID> id);
    void enqueue_erase(std::string vault, const RecordID& id);
    void undo(const std::vector<UndoEntry>& entries) noexcept;

    ElipsInstance* db_;
    std::vector<PendingOp> ops_;
    bool done_{false};
};

}  // namespace elips

#endif  // ELIPS_ELIPS_HPP
