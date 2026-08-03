#include "elips/storage/WAL.hpp"

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

#include "elips/domain/Errors.hpp"
#include "elips/storage/FileSync.hpp"
#include "elips/storage/Serialization.hpp"

namespace elips {
namespace {

constexpr std::uint32_t wal_magic = 0xE1105E01U;

// Serialize the body (everything the CRC covers, excluding the trailing CRC).
std::string encode_body(const WAL::Entry& entry) {
    std::ostringstream body(std::ios::binary);
    detail::put<std::uint32_t>(body, wal_magic);
    const bool has_extras = entry.document.has_value() || entry.chunk.has_value() ||
                            entry.lineage.has_value();
    // insert_q always carries its extras inline, so it is not promoted to
    // insert_ex the way a plain insert is.
    const auto op = (has_extras && entry.op != WAL::Op::insert_q)
                        ? WAL::Op::insert_ex
                        : entry.op;
    detail::put<std::uint8_t>(body, static_cast<std::uint8_t>(op));
    detail::put_string(body, entry.vault);
    body.write(reinterpret_cast<const char*>(entry.id.bytes().data()),
               static_cast<std::streamsize>(entry.id.bytes().size()));
    if (op == WAL::Op::insert_q) {
        detail::put<std::uint8_t>(body, static_cast<std::uint8_t>(entry.codec));
        detail::put<std::uint16_t>(
            body, static_cast<std::uint16_t>(entry.codes.size()));
        body.write(reinterpret_cast<const char*>(entry.codes.data()),
                   static_cast<std::streamsize>(entry.codes.size()));
        detail::put_payload(body, entry.payload);
        detail::put_document_attachment(body, entry.document);
        detail::put_chunk_info(body, entry.chunk);
        detail::put_embedding_lineage(body, entry.lineage);
        return body.str();
    }
    if (op == WAL::Op::insert || op == WAL::Op::insert_ex) {
        detail::put<std::uint16_t>(
            body, static_cast<std::uint16_t>(entry.vector.size()));
        body.write(reinterpret_cast<const char*>(entry.vector.data()),
                   static_cast<std::streamsize>(entry.vector.size() *
                                                sizeof(float)));
        detail::put_payload(body, entry.payload);
        if (op == WAL::Op::insert_ex) {
            detail::put_document_attachment(body, entry.document);
            detail::put_chunk_info(body, entry.chunk);
            detail::put_embedding_lineage(body, entry.lineage);
        }
    }
    return body.str();
}

}  // namespace

WAL::WAL(std::filesystem::path path, bool sync_each_write)
    : path_(std::move(path)), sync_each_write_(sync_each_write) {
#ifdef _WIN32
    fd_ = ::_open(path_.string().c_str(), _O_WRONLY | _O_CREAT | _O_APPEND | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    fd_ = ::open(path_.string().c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
#endif
    if (fd_ < 0) {
        throw StorageError{"cannot open WAL for appending"};
    }
}

WAL::~WAL() {
    if (fd_ >= 0) {
#ifdef _WIN32
        ::_close(fd_);
#else
        ::close(fd_);
#endif
    }
}

void WAL::write_all(const char* data, std::size_t size) {
    std::size_t written = 0;
    while (written < size) {
#ifdef _WIN32
        const int n = ::_write(fd_, data + written, static_cast<unsigned int>(size - written));
#else
        const std::ptrdiff_t n = ::write(fd_, data + written, size - written);
#endif
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw StorageError{"WAL append failed"};
        }
        written += static_cast<std::size_t>(n);
    }
}

void WAL::append(const Entry& entry) {
    const std::string body = encode_body(entry);
    const std::uint32_t crc = detail::crc32c(body.data(), body.size());
    // One write() per record so a crash can only ever truncate the tail, never
    // interleave two records.
    std::string record = body;
    record.append(reinterpret_cast<const char*>(&crc), sizeof(crc));
    write_all(record.data(), record.size());
    if (sync_each_write_) {
        detail::sync_file_data(fd_);  // reach stable storage before acknowledging
    }
}

void WAL::sync() {
    if (fd_ >= 0) {
        detail::sync_file_data(fd_);
    }
}

void WAL::append_insert(const std::string& vault, const RecordID& id,
                        std::span<const float> vector, const Payload& payload,
                        const std::optional<DocumentAttachment>& document,
                        const std::optional<ChunkInfo>& chunk,
                        const std::optional<EmbeddingLineage>& lineage) {
    append(Entry{Op::insert, vault, id,
                 std::vector<float>(vector.begin(), vector.end()), payload,
                 document, chunk, lineage, {}, quant::CodecId::none});
}

void WAL::append_insert_quantized(
    const std::string& vault, const RecordID& id,
    std::span<const std::uint8_t> codes, quant::CodecId codec,
    const Payload& payload, const std::optional<DocumentAttachment>& document,
    const std::optional<ChunkInfo>& chunk,
    const std::optional<EmbeddingLineage>& lineage) {
    append(Entry{Op::insert_q, vault, id, {}, payload, document, chunk, lineage,
                 std::vector<std::uint8_t>(codes.begin(), codes.end()), codec});
}

void WAL::append_erase(const std::string& vault, const RecordID& id) {
    append(Entry{Op::erase, vault, id, {}, {}, std::nullopt, std::nullopt,
                 std::nullopt, {}, quant::CodecId::none});
}

void WAL::append_txn_begin() {
    append(Entry{Op::txn_begin, {}, RecordID{}, {}, {}, std::nullopt,
                 std::nullopt, std::nullopt, {}, quant::CodecId::none});
}

void WAL::append_txn_commit() {
    append(Entry{Op::txn_commit, {}, RecordID{}, {}, {}, std::nullopt,
                 std::nullopt, std::nullopt, {}, quant::CodecId::none});
}

void WAL::reset() {
    if (fd_ >= 0) {
#ifdef _WIN32
        ::_close(fd_);
#else
        ::close(fd_);
#endif
    }
#ifdef _WIN32
    fd_ = ::_open(path_.string().c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    fd_ = ::open(path_.string().c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
    if (fd_ < 0) {
        throw StorageError{"cannot truncate WAL"};
    }
    detail::sync_file_data(fd_);
    detail::sync_directory(path_.parent_path());
}

namespace {

// Read-only streambuf over an existing buffer: lets replay() parse each record
// in place instead of copying the remaining log tail per iteration (which made
// replay O(n^2) in record count).
class SpanBuf final : public std::streambuf {
public:
    SpanBuf(const char* data, std::size_t size) {
        auto* base = const_cast<char*>(data);
        setg(base, base, base + size);
    }

protected:
    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode which) override {
        if ((which & std::ios_base::in) == 0) {
            return pos_type(off_type(-1));
        }
        off_type target = off;
        if (dir == std::ios_base::cur) {
            target += gptr() - eback();
        } else if (dir == std::ios_base::end) {
            target += egptr() - eback();
        }
        if (target < 0 || target > egptr() - eback()) {
            return pos_type(off_type(-1));
        }
        setg(eback(), eback() + target, egptr());
        return pos_type(target);
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(off_type(pos), std::ios_base::beg, which);
    }
};

}  // namespace

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
        SpanBuf buf(blob.data() + record_start, n - record_start);
        std::istream body(&buf);
        // Re-parse from the record start using a stream view.
        body.seekg(static_cast<std::streamoff>(4));  // skip magic (validated)
        Entry entry;
        try {
            const auto op = static_cast<Op>(detail::get<std::uint8_t>(body));
            entry.op = op;
            entry.vault = detail::get_string(body);
            RecordID::Bytes id_bytes{};
            body.read(reinterpret_cast<char*>(id_bytes.data()),
                      static_cast<std::streamsize>(id_bytes.size()));
            entry.id = RecordID{id_bytes};
            if (op == Op::insert_q) {
                entry.codec =
                    static_cast<quant::CodecId>(detail::get<std::uint8_t>(body));
                const auto code_len = detail::get<std::uint16_t>(body);
                // Bound the length against what is left before allocating: this
                // is replay, reachable from a hostile or corrupt log ahead of
                // the CRC check that is meant to reject it.
                detail::check_length(body, code_len);
                entry.codes.resize(code_len);
                body.read(reinterpret_cast<char*>(entry.codes.data()),
                          static_cast<std::streamsize>(code_len));
                entry.payload = detail::get_payload(body);
                entry.document = detail::get_document_attachment(body);
                entry.chunk = detail::get_chunk_info(body);
                entry.lineage = detail::get_embedding_lineage(body);
            } else if (op == Op::insert || op == Op::insert_ex) {
                const auto dim = detail::get<std::uint16_t>(body);
                detail::check_length(
                    body, static_cast<std::uint64_t>(dim) * sizeof(float));
                entry.vector.resize(dim);
                body.read(reinterpret_cast<char*>(entry.vector.data()),
                          static_cast<std::streamsize>(static_cast<std::size_t>(dim) * sizeof(float)));
                entry.payload = detail::get_payload(body);
                if (op == Op::insert_ex) {
                    entry.document = detail::get_document_attachment(body);
                    entry.chunk = detail::get_chunk_info(body);
                    entry.lineage = detail::get_embedding_lineage(body);
                }
            }
        } catch (const StorageError&) {
            break;  // malformed record: treat like a corrupt tail
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

    // Apply transaction framing: ops inside an unterminated txn_begin..(no
    // commit) window never happened as far as recovery is concerned.
    std::vector<Entry> applied;
    applied.reserve(entries.size());
    std::vector<Entry> pending;
    bool in_txn = false;
    for (auto& entry : entries) {
        switch (entry.op) {
            case Op::txn_begin:
                in_txn = true;
                pending.clear();
                break;
            case Op::txn_commit:
                for (auto& op : pending) {
                    applied.push_back(std::move(op));
                }
                pending.clear();
                in_txn = false;
                break;
            default:
                if (in_txn) {
                    pending.push_back(std::move(entry));
                } else {
                    applied.push_back(std::move(entry));
                }
                break;
        }
    }
    return applied;
}

}  // namespace elips
