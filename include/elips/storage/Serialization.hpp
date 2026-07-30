#ifndef ELIPS_STORAGE_SERIALIZATION_HPP
#define ELIPS_STORAGE_SERIALIZATION_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>
#include <variant>

#include "elips/domain/Errors.hpp"
#include "elips/domain/Record.hpp"

// Internal binary (de)serialization primitives shared by the snapshot and WAL.
// Native byte order: single-machine embedded use (cross-platform normalization
// is a later hardening item). Not part of the public API.
namespace elips::detail {

template <typename T>
void put(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T get(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

// Bytes left before end-of-stream, or -1 when the stream is not seekable (in
// which case callers fall back to the hard cap below).
inline std::streamoff stream_remaining(std::istream& in) {
    if (!in) {
        return 0;
    }
    const std::streampos cur = in.tellg();
    if (cur < 0) {
        return -1;
    }
    in.seekg(0, std::ios::end);
    const std::streampos end = in.tellg();
    in.seekg(cur, std::ios::beg);
    if (end < cur) {
        return 0;
    }
    return end - cur;
}

// Ceiling for a single length-prefixed field on a non-seekable stream. Any
// legitimate string/blob in this format is far smaller.
constexpr std::uint64_t max_field_bytes = 64ULL * 1024 * 1024;

// Reject a length prefix that cannot possibly be satisfied by the remaining
// bytes, *before* allocating for it. Corrupt or hostile input would otherwise
// request up to 4 GiB (or spin a 4-billion-iteration loop) ahead of the CRC
// check that is supposed to catch it.
inline void check_length(std::istream& in, std::uint64_t bytes_needed) {
    const std::streamoff avail = stream_remaining(in);
    if (avail < 0) {
        if (bytes_needed > max_field_bytes) {
            throw StorageError{"length prefix exceeds maximum field size"};
        }
        return;
    }
    if (bytes_needed > static_cast<std::uint64_t>(avail)) {
        throw StorageError{"length prefix exceeds remaining input"};
    }
}

inline void put_string(std::ostream& out, const std::string& s) {
    put<std::uint32_t>(out, static_cast<std::uint32_t>(s.size()));
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

inline std::string get_string(std::istream& in) {
    const auto len = get<std::uint32_t>(in);
    check_length(in, len);
    std::string s(len, '\0');
    in.read(s.data(), static_cast<std::streamsize>(len));
    if (!in) {
        throw StorageError{"truncated string field"};
    }
    return s;
}

inline void put_payload(std::ostream& out, const Payload& payload) {
    put<std::uint32_t>(out, static_cast<std::uint32_t>(payload.size()));
    for (const auto& [key, value] : payload) {
        put_string(out, key);
        put<std::uint8_t>(out, static_cast<std::uint8_t>(value.index()));
        std::visit(
            [&out](const auto& v) {
                using V = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<V, std::string>) {
                    put_string(out, v);
                } else {
                    put<V>(out, v);
                }
            },
            value);
    }
}

inline Payload get_payload(std::istream& in) {
    Payload payload;
    const auto count = get<std::uint32_t>(in);
    // Cheapest possible entry is 4-byte key length + 1-byte tag + 1-byte value.
    check_length(in, static_cast<std::uint64_t>(count) * 6ULL);
    for (std::uint32_t i = 0; i < count; ++i) {
        if (!in) {
            throw StorageError{"truncated payload"};
        }
        std::string key = get_string(in);
        const auto tag = get<std::uint8_t>(in);
        switch (tag) {
            case 0:
                payload.emplace(std::move(key), get<std::int64_t>(in));
                break;
            case 1:
                payload.emplace(std::move(key), get<double>(in));
                break;
            case 2:
                payload.emplace(std::move(key), get<bool>(in));
                break;
            case 3:
                payload.emplace(std::move(key), get_string(in));
                break;
            default:
                throw StorageError{"unknown payload value type tag"};
        }
    }
    if (!in) {
        throw StorageError{"truncated payload"};
    }
    return payload;
}

inline void put_document_attachment(
    std::ostream& out, const std::optional<DocumentAttachment>& document) {
    put<std::uint8_t>(out, document.has_value() ? 1U : 0U);
    if (!document.has_value()) {
        return;
    }
    put_string(out, document->text);
    put_string(out, document->uri);
    put_string(out, document->mime_type);
}

inline std::optional<DocumentAttachment> get_document_attachment(std::istream& in) {
    if (get<std::uint8_t>(in) == 0U) {
        return std::nullopt;
    }
    return DocumentAttachment{
        .text = get_string(in),
        .uri = get_string(in),
        .mime_type = get_string(in),
    };
}

inline void put_chunk_info(std::ostream& out, const std::optional<ChunkInfo>& chunk) {
    put<std::uint8_t>(out, chunk.has_value() ? 1U : 0U);
    if (!chunk.has_value()) {
        return;
    }
    put_string(out, chunk->document_key);
    put<std::uint32_t>(out, chunk->ordinal);
    put<std::uint32_t>(out, chunk->char_start);
    put<std::uint32_t>(out, chunk->char_end);
}

inline std::optional<ChunkInfo> get_chunk_info(std::istream& in) {
    if (get<std::uint8_t>(in) == 0U) {
        return std::nullopt;
    }
    return ChunkInfo{
        .document_key = get_string(in),
        .ordinal = get<std::uint32_t>(in),
        .char_start = get<std::uint32_t>(in),
        .char_end = get<std::uint32_t>(in),
    };
}

inline void put_embedding_lineage(
    std::ostream& out, const std::optional<EmbeddingLineage>& lineage) {
    put<std::uint8_t>(out, lineage.has_value() ? 1U : 0U);
    if (!lineage.has_value()) {
        return;
    }
    put_string(out, lineage->provider);
    put_string(out, lineage->model);
    put_string(out, lineage->revision);
    put_payload(out, lineage->attributes);
}

inline std::optional<EmbeddingLineage> get_embedding_lineage(std::istream& in) {
    if (get<std::uint8_t>(in) == 0U) {
        return std::nullopt;
    }
    return EmbeddingLineage{
        .provider = get_string(in),
        .model = get_string(in),
        .revision = get_string(in),
        .attributes = get_payload(in),
    };
}

// CRC32C (Castagnoli), software table built once. Used for WAL record integrity.
inline std::uint32_t crc32c(const void* data, std::size_t len) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t crc = i;
            for (int j = 0; j < 8; ++j) {
                crc = (crc & 1U) ? (crc >> 1) ^ 0x82F63B78U : (crc >> 1);
            }
            t[i] = crc;
        }
        return t;
    }();
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::uint32_t crc = 0xFFFFFFFFU;
    for (std::size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ bytes[i]) & 0xFFU] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFU;
}

}  // namespace elips::detail

#endif  // ELIPS_STORAGE_SERIALIZATION_HPP
