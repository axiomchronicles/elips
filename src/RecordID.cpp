#include "elips/domain/RecordID.hpp"

#include <chrono>
#include <cstdio>
#include <random>

#include "elips/domain/Errors.hpp"

namespace elips {
namespace {

constexpr std::size_t uuid_byte_count = 16;
constexpr std::size_t uuid_hex_length = 32;

std::uint64_t random_bits() {
    static thread_local std::mt19937_64 engine{std::random_device{}()};
    return engine();
}

std::uint8_t hex_nibble(char c) {
    if (c >= '0' && c <= '9') {
        return static_cast<std::uint8_t>(c - '0');
    }
    if (c >= 'a' && c <= 'f') {
        return static_cast<std::uint8_t>(c - 'a' + 10);
    }
    if (c >= 'A' && c <= 'F') {
        return static_cast<std::uint8_t>(c - 'A' + 10);
    }
    throw ConfigError{"invalid hex digit in RecordID"};
}

}  // namespace

RecordID RecordID::generate() {
    using namespace std::chrono;
    const auto now_ms = static_cast<std::uint64_t>(
        duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

    Bytes bytes{};
    // 48-bit big-endian millisecond timestamp (time-ordered prefix).
    for (int i = 0; i < 6; ++i) {
        bytes[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((now_ms >> (40 - 8 * i)) & 0xFF);
    }

    const std::uint64_t rand_a = random_bits();
    const std::uint64_t rand_b = random_bits();
    for (int i = 0; i < 5; ++i) {
        bytes[6 + static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((rand_a >> (8 * i)) & 0xFF);
    }
    for (int i = 0; i < 5; ++i) {
        bytes[11 + static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((rand_b >> (8 * i)) & 0xFF);
    }

    bytes[6] = static_cast<std::uint8_t>(0x70 | (bytes[6] & 0x0F));  // version 7
    bytes[8] = static_cast<std::uint8_t>(0x80 | (bytes[8] & 0x3F));  // variant 10
    return RecordID{bytes};
}

std::string RecordID::to_string() const {
    // Canonical 8-4-4-4-12 hyphenated form.
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(36);
    for (std::size_t i = 0; i < uuid_byte_count; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            out.push_back('-');
        }
        out.push_back(hex[bytes_[i] >> 4]);
        out.push_back(hex[bytes_[i] & 0x0F]);
    }
    return out;
}

RecordID RecordID::from_string(const std::string& text) {
    std::string digits;
    digits.reserve(uuid_hex_length);
    for (const char c : text) {
        if (c != '-') {
            digits.push_back(c);
        }
    }
    if (digits.size() != uuid_hex_length) {
        throw ConfigError{"RecordID string must contain 32 hex digits"};
    }
    Bytes bytes{};
    for (std::size_t i = 0; i < uuid_byte_count; ++i) {
        bytes[i] = static_cast<std::uint8_t>((hex_nibble(digits[2 * i]) << 4) |
                                             hex_nibble(digits[2 * i + 1]));
    }
    return RecordID{bytes};
}

}  // namespace elips
