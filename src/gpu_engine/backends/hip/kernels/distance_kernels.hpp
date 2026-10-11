// ELIPS HIP/ROCm backend — internal kernel launch declarations.
//
// EXPERIMENTAL: this header is compiled ONLY by hipcc (it pulls in
// <hip/hip_runtime.h>). It is included exclusively by the HIP translation
// units (HipBackend.hip, kernels/cosine_fp32.hip) and never by the regular
// C++ compiler that builds GpuDeviceManager.cpp — that unit only sees the
// hip-type-free HipBackend.hpp.
#ifndef ELIPS_GPU_ENGINE_HIP_DISTANCE_KERNELS_HPP
#define ELIPS_GPU_ENGINE_HIP_DISTANCE_KERNELS_HPP

#ifdef ELIPS_HIP_ENABLED

#include <hip/hip_runtime.h>
#include <cstdint>

#include "elips/Config.hpp"  // elips::Metric

namespace elips::gpu::hip::kernels {

// Row sum-of-squares: out[i] = sum_d data[i*dim + d]^2  (n rows).
void launch_row_sqnorms(const float* data, float* out, uint32_t n, uint32_t dim,
                        hipStream_t stream);

// Convert raw dot products (laid out row-major as dots[q*nb + b]) into the
// final per-metric distance in place, mirroring the Metal/CUDA reference:
//   cosine      -> 1 - dot            (assumes L2-normalized inputs)
//   dot_product -> -dot
//   euclidean   -> sqrt(max(0, |q|^2 + |b|^2 - 2*dot))
// q_sqnorms / b_sqnorms are only read for the euclidean case.
void launch_finalize_distance(float* dots, const float* q_sqnorms,
                              const float* b_sqnorms, uint32_t nq, uint32_t nb,
                              elips::Metric metric, hipStream_t stream);

// Per-query top-k selection (k smallest). One block per query; deterministic
// (val, index) lexicographic ordering so ties match the CPU partial_sort.
// Supports k up to 1024 (one thread per output slot).
void launch_topk(const float* distances, uint32_t* indices_out,
                 float* values_out, uint32_t nq, uint32_t nb, uint32_t k,
                 hipStream_t stream);

}  // namespace elips::gpu::hip::kernels

#endif  // ELIPS_HIP_ENABLED
#endif  // ELIPS_GPU_ENGINE_HIP_DISTANCE_KERNELS_HPP
