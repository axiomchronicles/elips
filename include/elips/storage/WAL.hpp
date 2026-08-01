#ifndef ELIPS_STORAGE_WAL_HPP
#define ELIPS_STORAGE_WAL_HPP

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "elips/domain/Record.hpp"
#include "elips/domain/RecordID.hpp"

namespace elips {

// Write-ahead log: every mutation is appended and (unless durability is
// relaxed) fsync'd before it is acknowledged, so writes survive both a process
// crash and an OS crash / power loss before the next checkpoint. On open the log
// is replayed on top of the last snapshot. Truncated/corrupt tail records are
// detected via CRC32C and cleanly dropped (no partial apply).
class WAL {
public:
    enum class Op : std::uint8_t {
        insert = 1,
        erase = 3,
        insert_ex = 4,
        txn_begin = 5,
        txn_commit = 6,
        // A quantized insert: the body carries the code rather than fp32
        // components. Codebooks are never logged -- quantize() checkpoints the
        // codebook durably and only then resets the log, so every insert_q on
        // disk postdates a codebook that can decode it.
        insert_q = 7,
    };

    struct Entry {
        Op op{Op::insert};
        std::string vault;
        RecordID id;
        std::vector<float> vector;  // empty for erase and insert_q
        Payload payload;
        std::optional<DocumentAttachment> document;
        std::optional<ChunkInfo> chunk;
        std::optional<EmbeddingLineage> lineage;
        std::vector<std::uint8_t> codes;  // populated only for insert_q
        quant::CodecId codec{quant::CodecId::none};
    };

    explicit WAL(std::filesystem::path path, bool sync_each_write = true);
    ~WAL();

    WAL(const WAL&) = delete;
    WAL& operator=(const WAL&) = delete;
    WAL(WAL&&) = delete;
    WAL& operator=(WAL&&) = delete;

    void append_insert(const std::string& vault, const RecordID& id,
                       std::span<const float> vector, const Payload& payload,
                       const std::optional<DocumentAttachment>& document = std::nullopt,
                       const std::optional<ChunkInfo>& chunk = std::nullopt,
                       const std::optional<EmbeddingLineage>& lineage = std::nullopt);
    void append_insert_quantized(
        const std::string& vault, const RecordID& id,
        std::span<const std::uint8_t> codes, quant::CodecId codec,
        const Payload& payload,
        const std::optional<DocumentAttachment>& document = std::nullopt,
        const std::optional<ChunkInfo>& chunk = std::nullopt,
        const std::optional<EmbeddingLineage>& lineage = std::nullopt);
    void append_erase(const std::string& vault, const RecordID& id);

    // Transaction framing. Records between begin and commit are only applied on
    // replay if the matching commit marker is present, so a crash (or an I/O
    // failure) partway through a batch cannot resurrect half of it.
    void append_txn_begin();
    void append_txn_commit();

    // Force any buffered records to stable storage (used before checkpointing
    // and by relaxed durability on close).
    void sync();

    // Truncate the log (called after a checkpoint has durably captured state).
    void reset();

    // Replay all valid records from a log file (stops at first invalid record).
    [[nodiscard]] static std::vector<Entry> replay(
        const std::filesystem::path& path);

private:
    void append(const Entry& entry);
    void write_all(const char* data, std::size_t size);

    std::filesystem::path path_;
    int fd_{-1};
    bool sync_each_write_{true};
};

}  // namespace elips

#endif  // ELIPS_STORAGE_WAL_HPP
