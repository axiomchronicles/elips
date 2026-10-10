// ELIPS HIP/ROCm distance + k-selection kernels.
//
// EXPERIMENTAL — NOT HARDWARE VALIDATED. Compiled with hipcc for the amdgpu
// platform. These mirror the CUDA kernels (HIP is CUDA-like); FAISS's own GPU
// code is ported to ROCm the same way (hipified from the CUDA sources).
//
// Compute strategy (FAISS-style): distances are produced by a hipBLAS SGEMM
// that yields raw dot products <q, b>, then these kernels finalize them into
// the requested metric. For L2 we use the identity
//   ||q - b||^2 = ||q||^2 + ||b||^2 - 2<q, b>
// with precomputed row squared-norms, which is why only a reconstruction pass
// is needed here rather than a full elementwise distance kernel.
#ifdef ELIPS_HIP_ENABLED

#include "distance_kernels.hpp"

namespace elips::gpu::hip::kernels {
namespace {

constexpr uint32_t kBlock = 256;

__global__ void row_sqnorms_kernel(const float* __restrict__ data,
                                   float* __restrict__ out, uint32_t n,
                                   uint32_t dim) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float acc = 0.0f;
    const float* r = data + static_cast<size_t>(row) * dim;
    for (uint32_t d = 0; d < dim; ++d) {
        const float v = r[d];
        acc += v * v;
    }
    out[row] = acc;
}

__global__ void finalize_cosine_kernel(float* __restrict__ dots, uint32_t total) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dots[i] = 1.0f - dots[i];
}

__global__ void finalize_dot_kernel(float* __restrict__ dots, uint32_t total) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dots[i] = -dots[i];
}

__global__ void finalize_l2_kernel(float* __restrict__ dots,
                                   const float* __restrict__ q_sq,
                                   const float* __restrict__ b_sq, uint32_t nq,
                                   uint32_t nb) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nq * nb) return;
    const uint32_t q = i / nb;
    const uint32_t b = i % nb;
    float s = q_sq[q] + b_sq[b] - 2.0f * dots[i];
    if (s < 0.0f) s = 0.0f;  // guard fp cancellation
    dots[i] = sqrtf(s);
}

// One thread per query. Maintains the running top-k (k smallest) as a sorted
// list directly in the output arrays via insertion, scanning all nb elements
// once: O(nb * k) per query. Deterministic (value, index) ordering matches the
// CPU std::partial_sort tie-breaking. Supports k up to 1024.
// ponytail: one-thread-per-query insertion selection; swap for a blockwise
// WarpSelect/bitonic if k-selection throughput ever matters on real hardware.
__global__ void topk_kernel(const float* __restrict__ distances,
                            uint32_t* __restrict__ indices_out,
                            float* __restrict__ values_out, uint32_t nq,
                            uint32_t nb, uint32_t k) {
    const uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    const float* row = distances + static_cast<size_t>(q) * nb;
    float* vals = values_out + static_cast<size_t>(q) * k;
    uint32_t* idxs = indices_out + static_cast<size_t>(q) * k;

    uint32_t filled = 0;
    for (uint32_t j = 0; j < nb; ++j) {
        const float v = row[j];
        if (filled < k) {
            // Insert into the sorted prefix [0, filled).
            uint32_t pos = filled;
            while (pos > 0 && vals[pos - 1] > v) {
                vals[pos] = vals[pos - 1];
                idxs[pos] = idxs[pos - 1];
                --pos;
            }
            vals[pos] = v;
            idxs[pos] = j;
            ++filled;
        } else if (v < vals[k - 1]) {
            uint32_t pos = k - 1;
            while (pos > 0 && vals[pos - 1] > v) {
                vals[pos] = vals[pos - 1];
                idxs[pos] = idxs[pos - 1];
                --pos;
            }
            vals[pos] = v;
            idxs[pos] = j;
        }
    }
}

inline uint32_t grid_for(uint32_t n) { return (n + kBlock - 1) / kBlock; }

}  // namespace

void launch_row_sqnorms(const float* data, float* out, uint32_t n, uint32_t dim,
                        hipStream_t stream) {
    if (n == 0) return;
    row_sqnorms_kernel<<<grid_for(n), kBlock, 0, stream>>>(data, out, n, dim);
}

void launch_finalize_distance(float* dots, const float* q_sqnorms,
                              const float* b_sqnorms, uint32_t nq, uint32_t nb,
                              elips::Metric metric, hipStream_t stream) {
    const uint32_t total = nq * nb;
    if (total == 0) return;
    const uint32_t grid = grid_for(total);
    switch (metric) {
        case elips::Metric::cosine:
            finalize_cosine_kernel<<<grid, kBlock, 0, stream>>>(dots, total);
            break;
        case elips::Metric::dot_product:
            finalize_dot_kernel<<<grid, kBlock, 0, stream>>>(dots, total);
            break;
        case elips::Metric::euclidean:
            finalize_l2_kernel<<<grid, kBlock, 0, stream>>>(dots, q_sqnorms,
                                                            b_sqnorms, nq, nb);
            break;
    }
}

void launch_topk(const float* distances, uint32_t* indices_out,
                 float* values_out, uint32_t nq, uint32_t nb, uint32_t k,
                 hipStream_t stream) {
    if (nq == 0 || k == 0) return;
    topk_kernel<<<grid_for(nq), kBlock, 0, stream>>>(distances, indices_out,
                                                     values_out, nq, nb, k);
}

}  // namespace elips::gpu::hip::kernels

#endif  // ELIPS_HIP_ENABLED
