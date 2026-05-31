#ifndef ELIPS_STORAGE_WAL_HPP
#define ELIPS_STORAGE_WAL_HPP

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "elips/domain/Record.hpp"
#include "elips/domain/RecordID.hpp"

namespace elips {

// Write-ahead log: every mutation is appended (and flushed) before it is
// acknowledged, so writes survive a crash before the next checkpoint. On open
// the log is replayed on top of the last snapshot. Truncated/corrupt tail
// records are detected via CRC32C and cleanly dropped (no partial apply).
class WAL {
public:
    enum class Op : std::uint8_t { insert = 1, erase = 3 };

    struct Entry {
        Op op{Op::insert};
        std::string vault;
        RecordID id;
        std::vector<float> vector;  // empty for erase
        Payload payload;
    };

    explicit WAL(std::filesystem::path path, bool sync_each_write = true);

    void append_insert(const std::string& vault, const RecordID& id,
                       std::span<const float> vector, const Payload& payload);
    void append_erase(const std::string& vault, const RecordID& id);

    // Truncate the log (called after a checkpoint has durably captured state).
    void reset();

    // Replay all valid records from a log file (stops at first invalid record).
    [[nodiscard]] static std::vector<Entry> replay(
        const std::filesystem::path& path);

private:
    void append(const Entry& entry);

    std::filesystem::path path_;
    std::ofstream out_;
    bool sync_each_write_{true};
};

}  // namespace elips

#endif  // ELIPS_STORAGE_WAL_HPP
