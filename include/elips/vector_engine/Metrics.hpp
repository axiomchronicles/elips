#ifndef ELIPS_VECTOR_ENGINE_METRICS_HPP
#define ELIPS_VECTOR_ENGINE_METRICS_HPP

#include <span>

#include "elips/Config.hpp"

namespace elips {

// Ordering-normalized distance: SMALLER means MORE similar for every metric,
// so callers can always sort ascending and keep the smallest k.
//   cosine     -> 1 - dot(a, b)        (inputs assumed L2-normalized on ingest)
//   euclidean  -> sqrt(sum (a-b)^2)
//   dot_product-> -dot(a, b)           (negated so larger dot sorts first)
// Scalar reference kernel; SIMD dispatch is a later-phase optimization (Per.2).
[[nodiscard]] float distance(Metric metric, std::span<const float> a,
                             std::span<const float> b) noexcept;

// Whether ingested/query vectors should be L2-normalized for this metric.
[[nodiscard]] bool requires_normalization(Metric metric) noexcept;

}  // namespace elips

#endif  // ELIPS_VECTOR_ENGINE_METRICS_HPP
