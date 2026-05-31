#ifndef ELIPS_DOMAIN_VECTOR_HPP
#define ELIPS_DOMAIN_VECTOR_HPP

#include <cmath>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace elips {

// Core domain type: an owned float32 vector with rich value semantics.
// Not a raw float array (I.4: precisely typed interfaces).
class Vector {
public:
    Vector() = default;
    explicit Vector(std::vector<float> values) : values_(std::move(values)) {}

    [[nodiscard]] std::span<const float> values() const noexcept { return values_; }
    [[nodiscard]] std::uint16_t dimension() const noexcept {
        return static_cast<std::uint16_t>(values_.size());
    }
    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }

    [[nodiscard]] float magnitude() const noexcept {
        float sum = 0.0F;
        for (const float v : values_) {
            sum += v * v;
        }
        return std::sqrt(sum);
    }

    // Returns an L2-normalized copy. Zero vectors are returned unchanged.
    [[nodiscard]] Vector normalized() const {
        const float mag = magnitude();
        if (mag == 0.0F) {
            return *this;
        }
        std::vector<float> out(values_.size());
        for (std::size_t i = 0; i < values_.size(); ++i) {
            out[i] = values_[i] / mag;
        }
        return Vector{std::move(out)};
    }

private:
    std::vector<float> values_;
};

}  // namespace elips

#endif  // ELIPS_DOMAIN_VECTOR_HPP
