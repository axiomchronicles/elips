// ============================================================================
// ELIPS CUDA kernels — host-callable launch wrappers.  EXPERIMENTAL.
// ============================================================================
#ifndef ELIPS_GPU_ENGINE_CUDA_KERNELS_CUH
#define ELIPS_GPU_ENGINE_CUDA_KERNELS_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace elips::gpu::cuda::kernels {

// Metric codes mirror elips::Metric declaration order.
enum MetricCode : int { kCosine = 0, kEuclidean = 1, kDotProduct = 2 };

// Row-wise squared L2 norms: norms[r] = sum_d data[r*dim + d]^2.
void launch_l2_norms_sq(const float* data, float* norms, int rows, int dim,
                        cudaStream_t stream);

// In-place convert a row-major (nq x nb) matrix of raw dot products into
// metric distances. For euclidean, q_norms (nq) and db_norms (nb) must hold
// squared L2 norms; they are ignored for cosine / dot_product.
void launch_finalize_distances(float* dist, const float* q_norms,
                               const float* db_norms, int nq, int nb,
                               int metric, cudaStream_t stream);

// Device k-selection: for each of nq rows of the (nq x nb) distances matrix,
// write the k smallest (value,index) pairs ascending into out_vals / out_idx
// (row stride k). Requires k <= nb.
void launch_topk(const float* distances, uint32_t* out_idx, float* out_vals,
                 int nq, int nb, int k, cudaStream_t stream);

} // namespace elips::gpu::cuda::kernels

#endif
