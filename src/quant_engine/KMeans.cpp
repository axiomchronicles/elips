#include "elips/quant_engine/detail/KMeans.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace elips::quant::detail {
namespace {

float squared_distance(const float* a, const float* b, std::size_t n) noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// C = A^T * B for row-major A (n x dim) and B (n x dim), yielding dim x dim.
std::vector<float> gram(std::span<const float> a_rows,
                        std::span<const float> b_rows, std::size_t n,
                        std::size_t dim) {
    std::vector<float> out(dim * dim, 0.0F);
    for (std::size_t row = 0; row < n; ++row) {
        const float* a = a_rows.data() + (row * dim);
        const float* b = b_rows.data() + (row * dim);
        for (std::size_t i = 0; i < dim; ++i) {
            const float scale = a[i];
            if (scale == 0.0F) {
                continue;
            }
            float* out_row = out.data() + (i * dim);
            for (std::size_t j = 0; j < dim; ++j) {
                out_row[j] += scale * b[j];
            }
        }
    }
    return out;
}

// C = A * B, all row-major dim x dim.
std::vector<float> matmul(std::span<const float> a, std::span<const float> b,
                          std::size_t dim) {
    std::vector<float> out(dim * dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) {
        float* out_row = out.data() + (i * dim);
        for (std::size_t k = 0; k < dim; ++k) {
            const float scale = a[(i * dim) + k];
            if (scale == 0.0F) {
                continue;
            }
            const float* b_row = b.data() + (k * dim);
            for (std::size_t j = 0; j < dim; ++j) {
                out_row[j] += scale * b_row[j];
            }
        }
    }
    return out;
}

std::vector<float> transpose(std::span<const float> m, std::size_t dim) {
    std::vector<float> out(dim * dim);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            out[(j * dim) + i] = m[(i * dim) + j];
        }
    }
    return out;
}

// Largest absolute row sum, an upper bound on the spectral norm. Used to scale
// M into the Newton-Schulz convergence basin.
float norm_bound(std::span<const float> m, std::size_t dim) noexcept {
    float best = 0.0F;
    for (std::size_t i = 0; i < dim; ++i) {
        float sum = 0.0F;
        for (std::size_t j = 0; j < dim; ++j) {
            sum += std::fabs(m[(i * dim) + j]);
        }
        best = std::max(best, sum);
    }
    return best;
}

bool all_finite(std::span<const float> values) noexcept {
    return std::all_of(values.begin(), values.end(),
                       [](float v) { return std::isfinite(v); });
}

// Forces a matrix to be exactly orthonormal via modified Gram-Schmidt over its
// rows.
//
// Necessary because Newton-Schulz only converges to the orthogonal polar factor
// when M has full rank. Real training sets routinely do not: a dimension with
// near-zero variance leaves M near-singular in that direction and the iteration
// shrinks the corresponding row toward zero instead of unit length. Decode
// applies the transpose as the inverse, so a row of norm 0.6 does not merely
// lose accuracy -- it scales that part of every reconstruction by 0.6 and the
// error is invisible until recall is measured.
//
// A row that collapses is replaced by the first canonical direction still
// orthogonal to what has been fixed so far, which yields a valid rotation that
// simply does not rotate that (information-free) subspace.
void orthonormalize(std::span<float> matrix, std::size_t dim) noexcept {
    constexpr float min_norm = 1e-4F;

    for (std::size_t i = 0; i < dim; ++i) {
        float* row = matrix.data() + (i * dim);

        for (std::size_t pass = 0; pass < 2; ++pass) {
            // Twice: one sweep leaves residual overlap at float precision once
            // dim is large, which is exactly where this matters.
            for (std::size_t j = 0; j < i; ++j) {
                const float* fixed = matrix.data() + (j * dim);
                float projection = 0.0F;
                for (std::size_t k = 0; k < dim; ++k) {
                    projection += row[k] * fixed[k];
                }
                for (std::size_t k = 0; k < dim; ++k) {
                    row[k] -= projection * fixed[k];
                }
            }
        }

        float norm = 0.0F;
        for (std::size_t k = 0; k < dim; ++k) {
            norm += row[k] * row[k];
        }
        norm = std::sqrt(norm);

        if (norm < min_norm || !std::isfinite(norm)) {
            // Degenerate: rebuild from a canonical direction and re-project.
            for (std::size_t candidate = 0; candidate < dim; ++candidate) {
                std::fill_n(row, dim, 0.0F);
                row[candidate] = 1.0F;
                for (std::size_t j = 0; j < i; ++j) {
                    const float* fixed = matrix.data() + (j * dim);
                    const float projection = fixed[candidate];
                    for (std::size_t k = 0; k < dim; ++k) {
                        row[k] -= projection * fixed[k];
                    }
                }
                norm = 0.0F;
                for (std::size_t k = 0; k < dim; ++k) {
                    norm += row[k] * row[k];
                }
                norm = std::sqrt(norm);
                if (norm >= min_norm) {
                    break;
                }
            }
            if (norm < min_norm || !std::isfinite(norm)) {
                return;  // cannot happen for a square basis; leave as-is
            }
        }

        const float inv = 1.0F / norm;
        for (std::size_t k = 0; k < dim; ++k) {
            row[k] *= inv;
        }
    }
}

}  // namespace

std::uint8_t nearest_centroid(std::span<const float> subvector,
                              std::span<const float> subspace_codebook,
                              std::size_t sub_dim, std::size_t ksub) noexcept {
    float best_distance = std::numeric_limits<float>::max();
    std::size_t best = 0;
    for (std::size_t c = 0; c < ksub; ++c) {
        const float d = squared_distance(
            subvector.data(), subspace_codebook.data() + (c * sub_dim), sub_dim);
        if (d < best_distance) {
            best_distance = d;
            best = c;
        }
    }
    return static_cast<std::uint8_t>(best);
}

void kmeans(std::span<const float> rows, std::size_t n, std::size_t dim,
            std::size_t k, std::size_t iters, std::span<float> centroids_out) {
    if (n == 0 || dim == 0 || k == 0) {
        return;
    }

    // Deterministic strided seeding. With fewer rows than centroids the stride
    // collapses to 1 and the tail duplicates the last row; Lloyd's reseeding
    // below then spreads the duplicates back out.
    const std::size_t stride = std::max<std::size_t>(1, n / k);
    for (std::size_t c = 0; c < k; ++c) {
        const std::size_t sample = std::min(n - 1, c * stride);
        std::copy_n(rows.data() + (sample * dim), dim,
                    centroids_out.data() + (c * dim));
    }

    std::vector<float> sums(k * dim, 0.0F);
    std::vector<std::size_t> counts(k, 0);

    for (std::size_t iter = 0; iter < iters; ++iter) {
        std::fill(sums.begin(), sums.end(), 0.0F);
        std::fill(counts.begin(), counts.end(), 0);

        for (std::size_t row = 0; row < n; ++row) {
            const float* vector = rows.data() + (row * dim);
            float best_distance = std::numeric_limits<float>::max();
            std::size_t best = 0;
            for (std::size_t c = 0; c < k; ++c) {
                const float d = squared_distance(
                    vector, centroids_out.data() + (c * dim), dim);
                if (d < best_distance) {
                    best_distance = d;
                    best = c;
                }
            }
            ++counts[best];
            float* sum = sums.data() + (best * dim);
            for (std::size_t i = 0; i < dim; ++i) {
                sum[i] += vector[i];
            }
        }

        bool moved = false;
        for (std::size_t c = 0; c < k; ++c) {
            float* centroid = centroids_out.data() + (c * dim);
            if (counts[c] == 0) {
                // Reseed an empty cluster from a distinct row rather than
                // leaving a centroid that can never win an assignment.
                std::copy_n(rows.data() + ((c % n) * dim), dim, centroid);
                moved = true;
                continue;
            }
            const float inv = 1.0F / static_cast<float>(counts[c]);
            for (std::size_t i = 0; i < dim; ++i) {
                const float updated = sums[(c * dim) + i] * inv;
                if (updated != centroid[i]) {
                    moved = true;
                }
                centroid[i] = updated;
            }
        }
        if (!moved) {
            break;  // converged; further iterations cannot change the codebook
        }
    }
}

void train_pq_codebook(std::span<const float> rows, std::size_t n,
                       std::size_t dim, std::size_t pq_dim, std::size_t ksub,
                       std::size_t iters, std::span<float> codebook_out) {
    if (n == 0 || pq_dim == 0 || dim == 0 || dim % pq_dim != 0 || ksub == 0) {
        return;
    }
    const std::size_t sub_dim = dim / pq_dim;

    // Gather each subspace into a contiguous scratch buffer so kmeans() sees
    // dense rows rather than a strided view of the full vectors.
    std::vector<float> scratch(n * sub_dim);
    for (std::size_t m = 0; m < pq_dim; ++m) {
        for (std::size_t row = 0; row < n; ++row) {
            std::copy_n(rows.data() + (row * dim) + (m * sub_dim), sub_dim,
                        scratch.data() + (row * sub_dim));
        }
        kmeans(scratch, n, sub_dim, ksub, iters,
               codebook_out.subspan(m * ksub * sub_dim, ksub * sub_dim));
    }
}

void encode_pq(std::span<const float> rows, std::span<const float> codebook,
               std::size_t n, std::size_t dim, std::size_t pq_dim,
               std::size_t ksub, std::span<std::uint8_t> codes_out) {
    if (pq_dim == 0 || dim % pq_dim != 0) {
        return;
    }
    const std::size_t sub_dim = dim / pq_dim;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t m = 0; m < pq_dim; ++m) {
            codes_out[(row * pq_dim) + m] = nearest_centroid(
                rows.subspan((row * dim) + (m * sub_dim), sub_dim),
                codebook.subspan(m * ksub * sub_dim, ksub * sub_dim), sub_dim,
                ksub);
        }
    }
}

std::vector<float> identity_matrix(std::size_t dim) {
    std::vector<float> out(dim * dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) {
        out[(i * dim) + i] = 1.0F;
    }
    return out;
}

void apply_matrix(std::span<const float> matrix, std::span<const float> vector,
                  std::span<float> out, std::size_t dim) noexcept {
    for (std::size_t i = 0; i < dim; ++i) {
        const float* row = matrix.data() + (i * dim);
        float sum = 0.0F;
        for (std::size_t j = 0; j < dim; ++j) {
            sum += row[j] * vector[j];
        }
        out[i] = sum;
    }
}

void apply_matrix_transposed(std::span<const float> matrix,
                             std::span<const float> vector, std::span<float> out,
                             std::size_t dim) noexcept {
    std::fill_n(out.data(), dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) {
        const float scale = vector[i];
        if (scale == 0.0F) {
            continue;
        }
        const float* row = matrix.data() + (i * dim);
        for (std::size_t j = 0; j < dim; ++j) {
            out[j] += row[j] * scale;
        }
    }
}

std::vector<float> solve_procrustes(std::span<const float> x_rows,
                                    std::span<const float> y_rows,
                                    std::size_t n, std::size_t dim) {
    if (n == 0 || dim == 0) {
        return identity_matrix(dim);
    }

    // The minimizer of ||R X - Y||_F over orthonormal R is the orthogonal polar
    // factor of M = Y^T X. Rather than pull in LAPACK for one SVD, take the
    // polar factor directly via Newton-Schulz, which converges quadratically
    // for a matrix scaled into ||M|| < sqrt(3) and needs only matmuls.
    std::vector<float> m = gram(y_rows, x_rows, n, dim);

    const float bound = norm_bound(m, dim);
    if (!(bound > 0.0F) || !std::isfinite(bound)) {
        return identity_matrix(dim);
    }
    const float inv_scale = 1.0F / bound;
    for (float& value : m) {
        value *= inv_scale;
    }

    const std::vector<float> eye = identity_matrix(dim);
    constexpr std::size_t max_iters = 32;
    constexpr float convergence_epsilon = 1e-6F;

    for (std::size_t iter = 0; iter < max_iters; ++iter) {
        // M <- 0.5 * M * (3I - M^T M)
        const std::vector<float> mtm = matmul(transpose(m, dim), m, dim);
        std::vector<float> correction(dim * dim);
        for (std::size_t i = 0; i < dim * dim; ++i) {
            correction[i] = (3.0F * eye[i]) - mtm[i];
        }
        std::vector<float> next = matmul(m, correction, dim);
        for (float& value : next) {
            value *= 0.5F;
        }
        if (!all_finite(next)) {
            return identity_matrix(dim);
        }

        float delta = 0.0F;
        for (std::size_t i = 0; i < dim * dim; ++i) {
            delta = std::max(delta, std::fabs(next[i] - m[i]));
        }
        m = std::move(next);
        if (delta < convergence_epsilon) {
            orthonormalize(m, dim);
            return m;
        }
    }

    // Did not converge within the iteration budget, which means M was close to
    // singular: the training set spans fewer than `dim` directions. Whatever the
    // iteration produced is not orthonormal, so it is repaired rather than
    // returned -- decode() relies on the transpose being the exact inverse.
    if (!all_finite(m)) {
        return identity_matrix(dim);
    }
    orthonormalize(m, dim);
    return m;
}

}  // namespace elips::quant::detail
