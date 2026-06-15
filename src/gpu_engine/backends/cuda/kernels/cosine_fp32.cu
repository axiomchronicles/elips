// CUDA distance kernel stubs
// These kernels compile with nvcc and require the CUDA toolkit.
// They implement warp-parallel distance computation for fp32/fp16 vectors.

#ifdef ELIPS_CUDA_ENABLED
#error "CUDA kernels must be compiled with nvcc. See cuda/CMakeLists.txt."

/*
Expected implementation (cosine_fp32.cu):

extern "C" __global__ void cosine_distance_fp32(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ distances,
    int nq, int nb, int dim
) {
    int qi = blockIdx.x;
    int ni = blockIdx.y * blockDim.x + threadIdx.x;
    if (qi >= nq || ni >= nb) return;

    float dot = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float qv = queries[qi * dim + d];
        float dv = database[ni * dim + d];
        dot += qv * dv;
    }
    distances[qi * nb + ni] = 1.0f - dot;
}
*/
#endif
