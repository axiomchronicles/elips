// ============================================================================
// ELIPS CUDA GPU backend implementation  —  EXPERIMENTAL / NOT HARDWARE-VALIDATED
// ----------------------------------------------------------------------------
// Compiled by nvcc only when ELIPS_GPU_CUDA=ON. Mirrors the Metal reference
// backend: cuBLAS GEMM for batch distances, a device k-selection kernel, and
// cudaMalloc/cudaMemcpy buffer management. Build from source; not validated on
// real NVIDIA hardware. See CudaBackend.hpp banner.
// ============================================================================
#ifdef ELIPS_CUDA_ENABLED

#include "CudaBackend.hpp"
#include "kernels/elips_cuda_kernels.cuh"

#include <string>
#include <vector>

namespace elips::gpu::cuda {
namespace {

GpuError map_cuda_error(cudaError_t e) noexcept {
    switch (e) {
        case cudaErrorMemoryAllocation:     return GpuError::InsufficientMemory;
        case cudaErrorNoDevice:
        case cudaErrorInvalidDevice:
        case cudaErrorInsufficientDriver:   return GpuError::DeviceNotFound;
        case cudaErrorInitializationError:  return GpuError::InitializationFailed;
        default:                            return GpuError::KernelLaunchFailed;
    }
}

struct DeviceScratch {
    void* ptr{nullptr};
    cudaError_t alloc(size_t bytes) { return cudaMalloc(&ptr, bytes); }
    ~DeviceScratch() { if (ptr) cudaFree(ptr); }
};

} // namespace

CudaBackend::CudaBackend() = default;
CudaBackend::~CudaBackend() { shutdown(); }

std::expected<void, GpuError> CudaBackend::initialize(const GpuConfig& config) {
    int count = 0;
    if (cudaError_t e = cudaGetDeviceCount(&count); e != cudaSuccess || count == 0) {
        return std::unexpected(GpuError::DeviceNotFound);
    }
    device_index_ = (config.device_index >= 0 && config.device_index < count)
                        ? config.device_index : 0;
    if (cudaError_t e = cudaSetDevice(device_index_); e != cudaSuccess) {
        return std::unexpected(map_cuda_error(e));
    }

    cudaDeviceProp prop{};
    if (cudaError_t e = cudaGetDeviceProperties(&prop, device_index_); e != cudaSuccess) {
        return std::unexpected(map_cuda_error(e));
    }
    if (cudaError_t e = cudaStreamCreate(&stream_); e != cudaSuccess) {
        return std::unexpected(map_cuda_error(e));
    }
    if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
        return std::unexpected(GpuError::InitializationFailed);
    }
    cublasSetStream(cublas_, stream_);

    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    const int cc = prop.major * 10 + prop.minor;
    info_ = GpuDeviceInfo{};
    info_.name = std::string(prop.name) + " [" + kCudaMaturity + "]";
    info_.vendor = "NVIDIA";
    info_.backend = "cuda";
    info_.device_index = static_cast<uint32_t>(device_index_);
    info_.total_device_memory_bytes = total_bytes ? total_bytes : prop.totalGlobalMem;
    info_.free_device_memory_bytes = free_bytes;
    info_.has_unified_memory = false;
    info_.compute_capability_major = static_cast<uint32_t>(prop.major);
    info_.compute_capability_minor = static_cast<uint32_t>(prop.minor);
    info_.max_threads_per_block = static_cast<uint32_t>(prop.maxThreadsPerBlock);
    info_.multiprocessor_count = static_cast<uint32_t>(prop.multiProcessorCount);
    info_.shared_memory_per_block_bytes = prop.sharedMemPerBlock;
    info_.l2_cache_bytes = static_cast<size_t>(prop.l2CacheSize);
    info_.supports_fp16 = cc >= 60;
    info_.supports_bf16 = cc >= 80;
    info_.supports_int8 = cc >= 61;
    info_.supports_cagra = true;
    info_.supports_ivf_pq = true;
    info_.supports_dynamic_batching = true;
    info_.supports_half_precision_search = info_.supports_fp16;

    initialized_ = true;
    return {};
}

void CudaBackend::shutdown() noexcept {
    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    initialized_ = false;
}

GpuDeviceInfo CudaBackend::device_info() const noexcept { return info_; }
bool CudaBackend::is_available() const noexcept { return initialized_; }

std::expected<GpuBuffer, GpuError> CudaBackend::allocate_device(size_t bytes) {
    void* ptr = nullptr;
    if (cudaError_t e = cudaMalloc(&ptr, bytes); e != cudaSuccess) {
        return std::unexpected(map_cuda_error(e));
    }
    return GpuBuffer{ptr, bytes, ptr};
}

void CudaBackend::free_device(GpuBuffer buf) noexcept {
    if (buf.device_ptr()) cudaFree(buf.device_ptr());
}

std::expected<void, GpuError>
CudaBackend::upload(const void* host_src, GpuBuffer& dst, size_t bytes) {
    if (cudaError_t e = cudaMemcpy(dst.device_ptr(), host_src, bytes,
                                   cudaMemcpyHostToDevice); e != cudaSuccess) {
        return std::unexpected(GpuError::TransferFailed);
    }
    return {};
}

std::expected<void, GpuError>
CudaBackend::download(const GpuBuffer& src, void* host_dst, size_t bytes) {
    if (cudaError_t e = cudaMemcpy(host_dst, src.device_ptr(), bytes,
                                   cudaMemcpyDeviceToHost); e != cudaSuccess) {
        return std::unexpected(GpuError::TransferFailed);
    }
    return {};
}

// Host-in / host-out, matching the GpuBruteForceIndex contract: the raw dot
// products D = Q * DB^T are formed with cuBLAS, then a kernel finalizes them
// into the requested metric. distances_out is row-major (nq x nb).
std::expected<void, GpuError>
CudaBackend::compute_distances_batch(std::span<const float> queries,
                                     std::span<const float> database,
                                     std::span<float> distances_out, size_t nq, size_t nb,
                                     size_t dim, elips::Metric metric) {
    if (nq == 0 || nb == 0 || dim == 0) return {};

    DeviceScratch d_q, d_db, d_dot, d_qn, d_dbn;
    if (d_q.alloc(queries.size_bytes()) != cudaSuccess ||
        d_db.alloc(database.size_bytes()) != cudaSuccess ||
        d_dot.alloc(nq * nb * sizeof(float)) != cudaSuccess) {
        return std::unexpected(GpuError::InsufficientMemory);
    }
    if (cudaMemcpy(d_q.ptr, queries.data(), queries.size_bytes(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_db.ptr, database.data(), database.size_bytes(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return std::unexpected(GpuError::TransferFailed);
    }

    // Column-major cuBLAS: C(nb x nq) = DB(nb x dim) * Q(nq x dim)^T, with C's
    // (j,i) element landing at distances_out[i*nb + j] (row-major nq x nb).
    const float alpha = 1.0f, beta = 0.0f;
    const int m = static_cast<int>(nb), n = static_cast<int>(nq), kk = static_cast<int>(dim);
    if (cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, m, n, kk, &alpha,
                    static_cast<const float*>(d_db.ptr), kk,
                    static_cast<const float*>(d_q.ptr), kk, &beta,
                    static_cast<float*>(d_dot.ptr), m) != CUBLAS_STATUS_SUCCESS) {
        return std::unexpected(GpuError::KernelLaunchFailed);
    }

    const float* qn = nullptr;
    const float* dbn = nullptr;
    if (metric == elips::Metric::euclidean) {
        if (d_qn.alloc(nq * sizeof(float)) != cudaSuccess ||
            d_dbn.alloc(nb * sizeof(float)) != cudaSuccess) {
            return std::unexpected(GpuError::InsufficientMemory);
        }
        kernels::launch_l2_norms_sq(static_cast<const float*>(d_q.ptr),
                                    static_cast<float*>(d_qn.ptr), n, kk, stream_);
        kernels::launch_l2_norms_sq(static_cast<const float*>(d_db.ptr),
                                    static_cast<float*>(d_dbn.ptr), m, kk, stream_);
        qn = static_cast<const float*>(d_qn.ptr);
        dbn = static_cast<const float*>(d_dbn.ptr);
    }

    kernels::launch_finalize_distances(static_cast<float*>(d_dot.ptr), qn, dbn, n, m,
                                       static_cast<int>(metric), stream_);
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        return std::unexpected(GpuError::KernelLaunchFailed);
    }
    if (cudaMemcpy(distances_out.data(), d_dot.ptr, nq * nb * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return std::unexpected(GpuError::TransferFailed);
    }
    return {};
}

std::expected<void, GpuError>
CudaBackend::top_k(std::span<const float> distances, std::span<uint32_t> indices_out,
                   std::span<float> values_out, size_t nq, size_t nb, size_t k) {
    if (nq == 0 || nb == 0 || k == 0) return {};

    DeviceScratch d_dist, d_idx, d_val;
    if (d_dist.alloc(distances.size_bytes()) != cudaSuccess ||
        d_idx.alloc(nq * k * sizeof(uint32_t)) != cudaSuccess ||
        d_val.alloc(nq * k * sizeof(float)) != cudaSuccess) {
        return std::unexpected(GpuError::InsufficientMemory);
    }
    if (cudaMemcpy(d_dist.ptr, distances.data(), distances.size_bytes(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return std::unexpected(GpuError::TransferFailed);
    }

    kernels::launch_topk(static_cast<const float*>(d_dist.ptr),
                         static_cast<uint32_t*>(d_idx.ptr), static_cast<float*>(d_val.ptr),
                         static_cast<int>(nq), static_cast<int>(nb), static_cast<int>(k),
                         stream_);
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        return std::unexpected(GpuError::KernelLaunchFailed);
    }
    if (cudaMemcpy(indices_out.data(), d_idx.ptr, nq * k * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(values_out.data(), d_val.ptr, nq * k * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return std::unexpected(GpuError::TransferFailed);
    }
    return {};
}

void CudaBackend::synchronize() {
    if (stream_) cudaStreamSynchronize(stream_);
}

bool CudaBackend::is_idle() const noexcept {
    return !stream_ || cudaStreamQuery(stream_) == cudaSuccess;
}

} // namespace elips::gpu::cuda

#endif // ELIPS_CUDA_ENABLED
