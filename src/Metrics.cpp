#include "elips/vector_engine/Metrics.hpp"

#include <cmath>
#include <cstddef>

#include "elips/Config.hpp"
#include "elips/domain/Errors.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace elips {
namespace {

using KernelFn = float (*)(const float*, const float*, std::size_t) noexcept;

// ----------------------------- scalar fallback -----------------------------

[[maybe_unused]] float dot_scalar(const float* a, const float* b,
                                  std::size_t n) noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

[[maybe_unused]] float sql2_scalar(const float* a, const float* b,
                                   std::size_t n) noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#if defined(__ARM_NEON)

// ------------------------- ARM NEON (Apple Silicon) ------------------------

float dot_neon(const float* a, const float* b, std::size_t n) noexcept {
    float32x4_t acc = vdupq_n_f32(0.0F);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));  // FMA
    }
    float sum = vaddvq_f32(acc);
    for (; i < n; ++i) {  // scalar tail
        sum += a[i] * b[i];
    }
    return sum;
}

float sql2_neon(const float* a, const float* b, std::size_t n) noexcept {
    float32x4_t acc = vdupq_n_f32(0.0F);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc = vmlaq_f32(acc, d, d);
    }
    float sum = vaddvq_f32(acc);
    for (; i < n; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#endif

// Runtime dispatch: select the best kernel once. NEON is baseline on ARM; on
// x86 the same seam selects AVX2/AVX-512 via CPUID (future kernels).
struct Dispatch {
    KernelFn dot;
    KernelFn sql2;
    Dispatch() {
#if defined(__ARM_NEON)
        dot = &dot_neon;
        sql2 = &sql2_neon;
#else
        dot = &dot_scalar;
        sql2 = &sql2_scalar;
#endif
    }
};

const Dispatch& kernels() {
    static const Dispatch instance;
    return instance;
}

float dot(std::span<const float> a, std::span<const float> b) noexcept {
    return kernels().dot(a.data(), b.data(), a.size());
}

float squared_l2(std::span<const float> a, std::span<const float> b) noexcept {
    return kernels().sql2(a.data(), b.data(), a.size());
}

}  // namespace

float distance(Metric metric, std::span<const float> a,
               std::span<const float> b) noexcept {
    switch (metric) {
        case Metric::cosine:
            return 1.0F - dot(a, b);
        case Metric::euclidean:
            return std::sqrt(squared_l2(a, b));
        case Metric::dot_product:
            return -dot(a, b);
    }
    return 0.0F;
}

bool requires_normalization(Metric metric) noexcept {
    return metric == Metric::cosine;
}

std::string_view to_string(Metric metric) noexcept {
    switch (metric) {
        case Metric::cosine:
            return "cosine";
        case Metric::euclidean:
            return "euclidean";
        case Metric::dot_product:
            return "dot_product";
    }
    return "cosine";
}

Metric metric_from_string(std::string_view name) {
    if (name == "cosine") {
        return Metric::cosine;
    }
    if (name == "euclidean") {
        return Metric::euclidean;
    }
    if (name == "dot_product") {
        return Metric::dot_product;
    }
    throw ConfigError{"unknown metric name"};
}

}  // namespace elips
