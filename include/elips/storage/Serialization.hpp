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

inline void put_string(std::ostream& out, const std::string& s) {
    put<std::uint32_t>(out, static_cast<std::uint32_t>(s.size()));
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

inline std::string get_string(std::istream& in) {
    const auto len = get<std::uint32_t>(in);
    std::string s(len, '\0');
    in.read(s.data(), static_cast<std::streamsize>(len));
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
    for (std::uint32_t i = 0; i < count; ++i) {
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
    return payload;
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
