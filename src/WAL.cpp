#include "elips/storage/WAL.hpp"

#include <cstring>
#include <iterator>
#include <sstream>

#include "elips/domain/Errors.hpp"
#include "elips/storage/Serialization.hpp"

namespace elips {
namespace {

constexpr std::uint32_t wal_magic = 0xE1105E01U;

// Serialize the body (everything the CRC covers, excluding the trailing CRC).
std::string encode_body(const WAL::Entry& entry) {
    std::ostringstream body(std::ios::binary);
    detail::put<std::uint32_t>(body, wal_magic);
    detail::put<std::uint8_t>(body, static_cast<std::uint8_t>(entry.op));
    detail::put_string(body, entry.vault);
    body.write(reinterpret_cast<const char*>(entry.id.bytes().data()),
               static_cast<std::streamsize>(entry.id.bytes().size()));
    if (entry.op == WAL::Op::insert) {
        detail::put<std::uint16_t>(
            body, static_cast<std::uint16_t>(entry.vector.size()));
        body.write(reinterpret_cast<const char*>(entry.vector.data()),
                   static_cast<std::streamsize>(entry.vector.size() *
                                                sizeof(float)));
        detail::put_payload(body, entry.payload);
    }
    return body.str();
}

}  // namespace

WAL::WAL(std::filesystem::path path, bool sync_each_write)
    : path_(std::move(path)), sync_each_write_(sync_each_write) {
    out_.open(path_, std::ios::binary | std::ios::app);
    if (!out_) {
        throw StorageError{"cannot open WAL for appending"};
    }
}

void WAL::append(const Entry& entry) {
    const std::string body = encode_body(entry);
    const std::uint32_t crc = detail::crc32c(body.data(), body.size());
    out_.write(body.data(), static_cast<std::streamsize>(body.size()));
    detail::put<std::uint32_t>(out_, crc);
    if (sync_each_write_) {
        out_.flush();  // hand off to the OS before acknowledging the write
    }
    if (!out_) {
        throw StorageError{"WAL append failed"};
    }
}

void WAL::append_insert(const std::string& vault, const RecordID& id,
                        std::span<const float> vector, const Payload& payload) {
    append(Entry{Op::insert, vault, id,
                 std::vector<float>(vector.begin(), vector.end()), payload});
}

void WAL::append_erase(const std::string& vault, const RecordID& id) {
    append(Entry{Op::erase, vault, id, {}, {}});
}

void WAL::reset() {
    out_.close();
    out_.open(path_, std::ios::binary | std::ios::trunc);
    if (!out_) {
        throw StorageError{"cannot truncate WAL"};
    }
}

std::vector<WAL::Entry> WAL::replay(const std::filesystem::path& path) {
    std::vector<Entry> entries;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return entries;
    }
    // Read the whole log; we re-checksum each record against its stored CRC.
    const std::string blob((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    std::size_t pos = 0;
    const std::size_t n = blob.size();

    auto remaining = [&] { return n - pos; };
    auto read_u32 = [&](std::uint32_t& out) {
        if (remaining() < 4) {
            return false;
        }
        std::memcpy(&out, blob.data() + pos, 4);
        pos += 4;
        return true;
    };

    while (pos < n) {
        const std::size_t record_start = pos;
        std::uint32_t magic = 0;
        if (!read_u32(magic) || magic != wal_magic) {
            break;  // corrupt/truncated tail: stop cleanly
        }
        std::istringstream body(std::string(blob.data() + record_start, remaining() + 4),
                                std::ios::binary);
        // Re-parse from the record start using a stream view.
        body.seekg(static_cast<std::streamoff>(4));  // skip magic (validated)
        const auto op = static_cast<Op>(detail::get<std::uint8_t>(body));
        Entry entry;
        entry.op = op;
        entry.vault = detail::get_string(body);
        RecordID::Bytes id_bytes{};
        body.read(reinterpret_cast<char*>(id_bytes.data()),
                  static_cast<std::streamsize>(id_bytes.size()));
        entry.id = RecordID{id_bytes};
        if (op == Op::insert) {
            const auto dim = detail::get<std::uint16_t>(body);
            entry.vector.resize(dim);
            body.read(reinterpret_cast<char*>(entry.vector.data()),
                      static_cast<std::streamsize>(dim) * sizeof(float));
            entry.payload = detail::get_payload(body);
        }
        if (!body) {
            break;  // truncated record
        }
        const auto body_len = static_cast<std::size_t>(body.tellg());
        if (record_start + body_len + 4 > n) {
            break;  // CRC missing
        }
        std::uint32_t stored_crc = 0;
        std::memcpy(&stored_crc, blob.data() + record_start + body_len, 4);
        const std::uint32_t actual =
            detail::crc32c(blob.data() + record_start, body_len);
        if (stored_crc != actual) {
            break;  // checksum mismatch: stop cleanly
        }
        entries.push_back(std::move(entry));
        pos = record_start + body_len + 4;
    }
    return entries;
}

}  // namespace elips
