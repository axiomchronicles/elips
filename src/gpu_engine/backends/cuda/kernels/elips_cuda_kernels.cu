// ============================================================================
// ELIPS CUDA kernels  —  EXPERIMENTAL / NOT HARDWARE-VALIDATED
// ----------------------------------------------------------------------------
// Distance finalization + a device k-selection kernel.
//
// The k-selection is adapted (not copied) from the FAISS GPU block/warp-select
// design:
//   Copyright (c) Meta Platforms, Inc. and affiliates.
//   Licensed under the MIT License.
//   https://github.com/facebookresearch/faiss (faiss/gpu/utils/Select.cuh)
// FAISS keeps a per-thread register queue merged via warp shuffles; here we use
// a simpler one-block-per-query ascending threshold scan (see topk kernel).
// ============================================================================
#include "elips_cuda_kernels.cuh"

#include <cfloat>

namespace elips::gpu::cuda::kernels {
namespace {

__global__ void l2_norms_sq_kernel(const float* __restrict__ data,
                                   float* __restrict__ norms, int rows, int dim) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* row = data + static_cast<size_t>(r) * dim;
    float acc = 0.0f;
    for (int d = 0; d < dim; ++d) acc += row[d] * row[d];
    norms[r] = acc;
}

__global__ void finalize_kernel(float* __restrict__ dist, const float* __restrict__ q_norms,
                                const float* __restrict__ db_norms, int nq, int nb, int metric) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(nq) * nb;
    if (idx >= total) return;
    float dot = dist[idx];
    switch (metric) {
        case kCosine:     dist[idx] = 1.0f - dot; break;          // assumes normalized
        case kDotProduct: dist[idx] = -dot; break;
        case kEuclidean: {
            int i = static_cast<int>(idx / nb);
            int j = static_cast<int>(idx % nb);
            float v = q_norms[i] + db_norms[j] - 2.0f * dot;
            dist[idx] = sqrtf(v > 0.0f ? v : 0.0f);
            break;
        }
    }
}

// One block per query row. k iterations; each selects the next-smallest
// (value,index) strictly after the previous pick, via a block reduction.
// ponytail: O(k*nb) per query, k serial passes — correct and fully on-device,
// but a FAISS WarpSelect register-queue is the upgrade path if k*nb dominates.
__global__ void topk_kernel(const float* __restrict__ dist, uint32_t* __restrict__ out_idx,
                            float* __restrict__ out_val, int nb, int k) {
    extern __shared__ char smem[];
    float* s_val = reinterpret_cast<float*>(smem);
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    int q = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* row = dist + static_cast<size_t>(q) * nb;

    float prev_val = -FLT_MAX;
    int prev_idx = -1;

    for (int sel = 0; sel < k; ++sel) {
        float best = FLT_MAX;
        int best_i = -1;
        for (int j = tid; j < nb; j += nthreads) {
            float v = row[j];
            bool after = (v > prev_val) || (v == prev_val && j > prev_idx);
            if (!after) continue;
            if (best_i == -1 || v < best || (v == best && j < best_i)) {
                best = v;
                best_i = j;
            }
        }
        s_val[tid] = best;
        s_idx[tid] = best_i;
        __syncthreads();

        for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                int oi = s_idx[tid + stride];
                float ov = s_val[tid + stride];
                bool take = oi != -1 && (s_idx[tid] == -1 || ov < s_val[tid] ||
                                         (ov == s_val[tid] && oi < s_idx[tid]));
                if (take) {
                    s_val[tid] = ov;
                    s_idx[tid] = oi;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            out_val[static_cast<size_t>(q) * k + sel] = s_val[0];
            out_idx[static_cast<size_t>(q) * k + sel] = static_cast<uint32_t>(s_idx[0]);
        }
        prev_val = s_val[0];
        prev_idx = s_idx[0];
        __syncthreads();
    }
}

} // namespace

void launch_l2_norms_sq(const float* data, float* norms, int rows, int dim,
                        cudaStream_t stream) {
    int block = 256;
    int grid = (rows + block - 1) / block;
    l2_norms_sq_kernel<<<grid, block, 0, stream>>>(data, norms, rows, dim);
}

void launch_finalize_distances(float* dist, const float* q_norms, const float* db_norms,
                               int nq, int nb, int metric, cudaStream_t stream) {
    size_t total = static_cast<size_t>(nq) * nb;
    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    finalize_kernel<<<grid, block, 0, stream>>>(dist, q_norms, db_norms, nq, nb, metric);
}

void launch_topk(const float* distances, uint32_t* out_idx, float* out_vals, int nq, int nb,
                 int k, cudaStream_t stream) {
    int block = 256; // power of two for the shared-memory reduction
    size_t shmem = static_cast<size_t>(block) * (sizeof(float) + sizeof(int));
    topk_kernel<<<nq, block, shmem, stream>>>(distances, out_idx, out_vals, nb, k);
}

} // namespace elips::gpu::cuda::kernels
