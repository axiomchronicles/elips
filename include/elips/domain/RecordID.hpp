#ifndef ELIPS_DOMAIN_RECORD_ID_HPP
#define ELIPS_DOMAIN_RECORD_ID_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace elips {

// Stable, time-ordered record identity (UUIDv7). 128-bit value type.
class RecordID {
public:
    using Bytes = std::array<std::uint8_t, 16>;

    RecordID() = default;
    explicit RecordID(Bytes bytes) : bytes_(bytes) {}

    [[nodiscard]] const Bytes& bytes() const noexcept { return bytes_; }
    [[nodiscard]] std::string to_string() const;

    [[nodiscard]] static RecordID generate();  // UUIDv7, monotonic by time prefix
    [[nodiscard]] static RecordID from_string(const std::string& text);

    bool operator==(const RecordID&) const = default;
    // Lexicographic byte order == UUIDv7 time order (big-endian ms prefix).
    auto operator<=>(const RecordID&) const = default;

private:
    Bytes bytes_{};
};

}  // namespace elips

template <>
struct std::hash<elips::RecordID> {
    std::size_t operator()(const elips::RecordID& id) const noexcept {
        std::size_t h = 1469598103934665603ULL;  // FNV-1a
        for (const std::uint8_t b : id.bytes()) {
            h = (h ^ b) * 1099511628211ULL;
        }
        return h;
    }
};

#endif  // ELIPS_DOMAIN_RECORD_ID_HPP
