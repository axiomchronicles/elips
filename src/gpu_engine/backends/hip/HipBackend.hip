// ELIPS HIP/ROCm GPU backend — host implementation (compiled with hipcc).
//
// EXPERIMENTAL — NOT HARDWARE VALIDATED. Compile-checked only; there is no AMD
// GPU in CI. Mirrors the CUDA backend using HIP portability APIs plus hipBLAS
// SGEMM for batched distances and a device k-selection kernel. Build from
// source against a ROCm SDK and validate on real hardware before production
// use (same posture as FAISS's ROCm support, which is hipified from the CUDA
// code and shipped build-from-source rather than as a prebuilt package).
#ifdef ELIPS_HIP_ENABLED

#include "HipBackend.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <string>
#include <vector>

#include "kernels/distance_kernels.hpp"

namespace elips::gpu::hip {

namespace {

GpuError map_hip_error(hipError_t e) noexcept {
    switch (e) {
        case hipSuccess:              return GpuError::InitializationFailed;  // unused
        case hipErrorNoDevice:
        case hipErrorInvalidDevice:   return GpuError::DeviceNotFound;
        case hipErrorOutOfMemory:     return GpuError::InsufficientMemory;
        case hipErrorNotInitialized:  return GpuError::InitializationFailed;
        default:                      return GpuError::KernelLaunchFailed;
    }
}

GpuError map_blas_error(hipblasStatus_t s) noexcept {
    switch (s) {
        case HIPBLAS_STATUS_ALLOC_FAILED:    return GpuError::InsufficientMemory;
        case HIPBLAS_STATUS_NOT_INITIALIZED: return GpuError::InitializationFailed;
        default:                             return GpuError::KernelLaunchFailed;
    }
}

}  // namespace

struct HipBackend::Impl {
    int device_index{-1};
    bool initialized{false};
    GpuDeviceInfo info;
    hipStream_t stream{nullptr};
    hipblasHandle_t blas{nullptr};
};

HipBackend::HipBackend() : impl_(std::make_unique<Impl>()) {}
HipBackend::~HipBackend() { shutdown(); }

std::expected<void, GpuError> HipBackend::initialize(const GpuConfig& config) {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
        return std::unexpected(GpuError::DeviceNotFound);
    }

    const int idx = (config.device_index >= 0 && config.device_index < count)
                        ? config.device_index
                        : 0;
    if (hipSetDevice(idx) != hipSuccess) {
        return std::unexpected(GpuError::DeviceNotFound);
    }

    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, idx) != hipSuccess) {
        return std::unexpected(GpuError::InitializationFailed);
    }

    size_t free_bytes = 0, total_bytes = 0;
    static_cast<void>(hipMemGetInfo(&free_bytes, &total_bytes));

    auto& info = impl_->info;
    // EXPERIMENTAL maturity is surfaced in the device name since GpuDeviceInfo
    // has no dedicated maturity field (shared struct, left untouched).
    info.name = std::string(props.name) + " (EXPERIMENTAL - not hardware validated)";
    info.vendor = "AMD";
    info.backend = "hip";
    info.device_index = static_cast<uint32_t>(idx);
    info.total_device_memory_bytes = total_bytes;
    info.free_device_memory_bytes = free_bytes;
    info.has_unified_memory = false;  // discrete AMD GPU: separate device memory
    info.supports_fp16 = true;
    info.supports_bf16 = true;
    info.supports_int8 = true;
    info.compute_capability_major = static_cast<uint32_t>(props.major);
    info.compute_capability_minor = static_cast<uint32_t>(props.minor);
    info.max_threads_per_block = static_cast<uint32_t>(props.maxThreadsPerBlock);
    info.multiprocessor_count = static_cast<uint32_t>(props.multiProcessorCount);
    info.shared_memory_per_block_bytes = props.sharedMemPerBlock;
    info.l2_cache_bytes = static_cast<size_t>(props.l2CacheSize);
    info.supports_cagra = false;
    info.supports_ivf_pq = true;
    info.supports_dynamic_batching = true;
    info.supports_half_precision_search = true;

    if (hipStreamCreate(&impl_->stream) != hipSuccess) {
        return std::unexpected(GpuError::InitializationFailed);
    }
    if (hipblasCreate(&impl_->blas) != HIPBLAS_STATUS_SUCCESS) {
        static_cast<void>(hipStreamDestroy(impl_->stream));
        impl_->stream = nullptr;
        return std::unexpected(GpuError::InitializationFailed);
    }
    hipblasSetStream(impl_->blas, impl_->stream);

    impl_->device_index = idx;
    impl_->initialized = true;
    return {};
}

void HipBackend::shutdown() noexcept {
    if (impl_->blas) {
        hipblasDestroy(impl_->blas);
        impl_->blas = nullptr;
    }
    if (impl_->stream) {
        static_cast<void>(hipStreamDestroy(impl_->stream));
        impl_->stream = nullptr;
    }
    impl_->initialized = false;
}

GpuDeviceInfo HipBackend::device_info() const noexcept { return impl_->info; }
bool HipBackend::is_available() const noexcept { return impl_->initialized; }

std::expected<GpuBuffer, GpuError> HipBackend::allocate_device(size_t bytes) {
    void* ptr = nullptr;
    const hipError_t e = hipMalloc(&ptr, bytes);
    if (e != hipSuccess || ptr == nullptr) {
        return std::unexpected(map_hip_error(e));
    }
    // For HIP the device pointer is also the backend handle.
    return GpuBuffer{ptr, bytes, ptr};
}

void HipBackend::free_device(GpuBuffer buf) noexcept {
    if (buf.device_ptr()) {
        static_cast<void>(hipFree(buf.device_ptr()));
    }
}

std::expected<void, GpuError>
HipBackend::upload(const void* host_src, GpuBuffer& dst, size_t bytes) {
    const hipError_t e =
        hipMemcpy(dst.device_ptr(), host_src, bytes, hipMemcpyHostToDevice);
    if (e != hipSuccess) return std::unexpected(GpuError::TransferFailed);
    return {};
}

std::expected<void, GpuError>
HipBackend::download(const GpuBuffer& src, void* host_dst, size_t bytes) {
    const hipError_t e =
        hipMemcpy(host_dst, src.device_ptr(), bytes, hipMemcpyDeviceToHost);
    if (e != hipSuccess) return std::unexpected(GpuError::TransferFailed);
    return {};
}

namespace {
// RAII device scratch allocation so early returns never leak.
struct DevScratch {
    void* ptr{nullptr};
    hipError_t status{hipSuccess};
    explicit DevScratch(size_t bytes) { status = hipMalloc(&ptr, bytes); }
    DevScratch(const DevScratch&) = delete;
    DevScratch& operator=(const DevScratch&) = delete;
    ~DevScratch() { if (ptr) static_cast<void>(hipFree(ptr)); }
    [[nodiscard]] bool ok() const { return status == hipSuccess && ptr; }
    template <typename T> T* as() const { return static_cast<T*>(ptr); }
};
}  // namespace

std::expected<void, GpuError> HipBackend::compute_distances_batch(
    std::span<const float> queries, std::span<const float> database,
    std::span<float> distances_out, size_t nq, size_t nb, size_t dim,
    elips::Metric metric) {
    if (!impl_->initialized) return std::unexpected(GpuError::BackendUnavailable);
    if (nq == 0 || nb == 0 || dim == 0) return {};

    DevScratch d_q(nq * dim * sizeof(float));
    DevScratch d_db(nb * dim * sizeof(float));
    DevScratch d_dots(nq * nb * sizeof(float));
    if (!d_q.ok() || !d_db.ok() || !d_dots.ok()) {
        return std::unexpected(GpuError::InsufficientMemory);
    }

    if (hipMemcpyAsync(d_q.ptr, queries.data(), nq * dim * sizeof(float),
                       hipMemcpyHostToDevice, impl_->stream) != hipSuccess ||
        hipMemcpyAsync(d_db.ptr, database.data(), nb * dim * sizeof(float),
                       hipMemcpyHostToDevice, impl_->stream) != hipSuccess) {
        return std::unexpected(GpuError::TransferFailed);
    }

    // Raw dot products via hipBLAS SGEMM. hipBLAS is column-major; computing
    // C(nb x nq) = db(nb x dim) * queries(nq x dim)^T under column-major with
    // op(A)=T on the row-major db and op(B)=N on the row-major queries yields
    // dots laid out exactly as the row-major distances[q*nb + b].
    const float one = 1.0f, zero = 0.0f;
    const hipblasStatus_t gemm = hipblasSgemm(
        impl_->blas, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(nb),
        static_cast<int>(nq), static_cast<int>(dim), &one, d_db.as<float>(),
        static_cast<int>(dim), d_q.as<float>(), static_cast<int>(dim), &zero,
        d_dots.as<float>(), static_cast<int>(nb));
    if (gemm != HIPBLAS_STATUS_SUCCESS) return std::unexpected(map_blas_error(gemm));

    // Euclidean needs row squared-norms for ||q||^2 + ||b||^2 - 2<q,b>.
    DevScratch d_qsq(metric == elips::Metric::euclidean ? nq * sizeof(float) : 0);
    DevScratch d_bsq(metric == elips::Metric::euclidean ? nb * sizeof(float) : 0);
    if (metric == elips::Metric::euclidean) {
        if (!d_qsq.ok() || !d_bsq.ok()) {
            return std::unexpected(GpuError::InsufficientMemory);
        }
        kernels::launch_row_sqnorms(d_q.as<float>(), d_qsq.as<float>(),
                                    static_cast<uint32_t>(nq),
                                    static_cast<uint32_t>(dim), impl_->stream);
        kernels::launch_row_sqnorms(d_db.as<float>(), d_bsq.as<float>(),
                                    static_cast<uint32_t>(nb),
                                    static_cast<uint32_t>(dim), impl_->stream);
    }

    kernels::launch_finalize_distance(
        d_dots.as<float>(), d_qsq.as<float>(), d_bsq.as<float>(),
        static_cast<uint32_t>(nq), static_cast<uint32_t>(nb), metric,
        impl_->stream);

    if (hipMemcpyAsync(distances_out.data(), d_dots.ptr,
                       nq * nb * sizeof(float), hipMemcpyDeviceToHost,
                       impl_->stream) != hipSuccess) {
        return std::unexpected(GpuError::TransferFailed);
    }
    if (hipStreamSynchronize(impl_->stream) != hipSuccess) {
        return std::unexpected(GpuError::KernelLaunchFailed);
    }
    return {};
}

std::expected<void, GpuError>
HipBackend::top_k(std::span<const float> distances, std::span<uint32_t> indices_out,
                  std::span<float> values_out, size_t nq, size_t nb, size_t k) {
    if (!impl_->initialized) return std::unexpected(GpuError::BackendUnavailable);
    if (nq == 0 || nb == 0 || k == 0) return {};

    DevScratch d_dist(nq * nb * sizeof(float));
    DevScratch d_idx(nq * k * sizeof(uint32_t));
    DevScratch d_val(nq * k * sizeof(float));
    if (!d_dist.ok() || !d_idx.ok() || !d_val.ok()) {
        return std::unexpected(GpuError::InsufficientMemory);
    }

    if (hipMemcpyAsync(d_dist.ptr, distances.data(), nq * nb * sizeof(float),
                       hipMemcpyHostToDevice, impl_->stream) != hipSuccess) {
        return std::unexpected(GpuError::TransferFailed);
    }

    kernels::launch_topk(d_dist.as<float>(), d_idx.as<uint32_t>(),
                         d_val.as<float>(), static_cast<uint32_t>(nq),
                         static_cast<uint32_t>(nb), static_cast<uint32_t>(k),
                         impl_->stream);

    if (hipMemcpyAsync(indices_out.data(), d_idx.ptr, nq * k * sizeof(uint32_t),
                       hipMemcpyDeviceToHost, impl_->stream) != hipSuccess ||
        hipMemcpyAsync(values_out.data(), d_val.ptr, nq * k * sizeof(float),
                       hipMemcpyDeviceToHost, impl_->stream) != hipSuccess) {
        return std::unexpected(GpuError::TransferFailed);
    }
    if (hipStreamSynchronize(impl_->stream) != hipSuccess) {
        return std::unexpected(GpuError::KernelLaunchFailed);
    }
    return {};
}

void HipBackend::synchronize() {
    if (impl_->stream) static_cast<void>(hipStreamSynchronize(impl_->stream));
}

bool HipBackend::is_idle() const noexcept {
    if (!impl_->stream) return true;
    return hipStreamQuery(impl_->stream) == hipSuccess;
}

} // namespace elips::gpu::hip

#endif // ELIPS_HIP_ENABLED

