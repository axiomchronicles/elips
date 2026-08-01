#ifndef ELIPS_QUANT_ENGINE_DETAIL_KMEANS_HPP
#define ELIPS_QUANT_ENGINE_DETAIL_KMEANS_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Shared k-means machinery behind the product quantizer. Header-visible (rather
// than tucked into a .cpp) because the GPU engine's quantization pipeline
// delegates here instead of carrying its own copy.
namespace elips::quant::detail {

// Trains `k` centroids over `n` rows of `dim` floats and writes k * dim floats
// to `centroids_out`.
//
// Initialization is a deterministic strided sample rather than k-means++: the
// same input always yields the same codebook, which makes a retrained index
// byte-reproducible and keeps the trainer free of RNG state that would
// otherwise have to be serialized. Empty clusters are reseeded from a distinct
// input row rather than left degenerate.
void kmeans(std::span<const float> rows, std::size_t n, std::size_t dim,
            std::size_t k, std::size_t iters, std::span<float> centroids_out);

// Trains one codebook per subspace over row-major `rows`.
//
// `dim` must be divisible by `pq_dim`. Writes pq_dim * ksub * sub_dim floats to
// `codebook_out`, laid out subspace-major: subspace, then centroid, then
// component.
void train_pq_codebook(std::span<const float> rows, std::size_t n,
                       std::size_t dim, std::size_t pq_dim, std::size_t ksub,
                       std::size_t iters, std::span<float> codebook_out);

// Assigns each row to its nearest centroid per subspace, writing n * pq_dim
// codes. `ksub` must not exceed 256.
void encode_pq(std::span<const float> rows, std::span<const float> codebook,
               std::size_t n, std::size_t dim, std::size_t pq_dim,
               std::size_t ksub, std::span<std::uint8_t> codes_out);

// Nearest centroid within one subspace codebook of `ksub` entries.
[[nodiscard]] std::uint8_t nearest_centroid(
    std::span<const float> subvector, std::span<const float> subspace_codebook,
    std::size_t sub_dim, std::size_t ksub) noexcept;

// Orthogonal Procrustes: the orthonormal R minimizing ||R X - Y||_F, where X
// and Y hold `n` rows of `dim` floats.
//
// Solved as the orthogonal polar factor of M = Y^T X via an inverse-free
// Newton-Schulz iteration, which avoids taking a LAPACK dependency for a
// single decomposition. Returns dim * dim floats row-major, or the identity
// when the iteration fails to converge (a rank-deficient M, which happens when
// the training set spans fewer than `dim` directions).
[[nodiscard]] std::vector<float> solve_procrustes(std::span<const float> x_rows,
                                                  std::span<const float> y_rows,
                                                  std::size_t n,
                                                  std::size_t dim);

// Row-major dim x dim identity.
[[nodiscard]] std::vector<float> identity_matrix(std::size_t dim);

// out = M * v for a row-major dim x dim M. `out` must not alias `v`.
void apply_matrix(std::span<const float> matrix, std::span<const float> vector,
                  std::span<float> out, std::size_t dim) noexcept;

// out = M^T * v, the inverse of apply_matrix for orthonormal M.
void apply_matrix_transposed(std::span<const float> matrix,
                             std::span<const float> vector, std::span<float> out,
                             std::size_t dim) noexcept;

}  // namespace elips::quant::detail

#endif  // ELIPS_QUANT_ENGINE_DETAIL_KMEANS_HPP
