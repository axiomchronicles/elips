#ifdef ELIPS_METAL_ENABLED

// The GPU distance + top-k dispatch here drives Metal kernels adapted from
// FAISS faiss/gpu_metal/MetalDistance.metal / MetalKernels.mm, MIT (c) Meta
// Platforms. See THIRD_PARTY_NOTICES.md.

#include "MetalBackend.hpp"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <dlfcn.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>

#ifndef ELIPS_METALLIB_BUILD_PATH
#error "ELIPS_METALLIB_BUILD_PATH must be defined by CMake"
#endif

namespace elips::gpu::metal {

struct MetalBackend::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    GpuDeviceInfo info;
    bool initialized{false};
    bool has_unified_memory{false};

    id<MTLComputePipelineState> pipeline(const std::string& name) {
        auto it = pipelines.find(name);
        if (it != pipelines.end()) return it->second;
        id<MTLFunction> fn = [library newFunctionWithName:@(name.c_str())];
        if (!fn) return nil;
        NSError* err = nil;
        id<MTLComputePipelineState> ps =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) return nil;
        pipelines[name] = ps;
        return ps;
    }
};

// Locate elips_kernels.metallib at runtime:
//   1. ELIPS_METALLIB_PATH env override
//   2. compiled-in build-tree path (dev builds)
//   3. next to the loaded library/binary via dladdr (installed packages)
static NSString* findMetalLibPath() {
    NSFileManager* fm = [NSFileManager defaultManager];

    const char* envPath = getenv("ELIPS_METALLIB_PATH");
    if (envPath) {
        NSString* ep = @(envPath);
        if ([fm fileExistsAtPath:ep]) return ep;
    }

    NSString* buildPath = @(ELIPS_METALLIB_BUILD_PATH);
    if ([fm fileExistsAtPath:buildPath]) return buildPath;

    Dl_info dl;
    if (dladdr(reinterpret_cast<const void*>(&findMetalLibPath), &dl) && dl.dli_fname) {
        NSString* dir = [@(dl.dli_fname) stringByDeletingLastPathComponent];
        NSString* installed =
            [dir stringByAppendingPathComponent:@"elips_kernels.metallib"];
        if ([fm fileExistsAtPath:installed]) return installed;
    }
    return nil;
}

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {
    impl_->device = nil;
    impl_->queue = nil;
    impl_->library = nil;
}

MetalBackend::~MetalBackend() { shutdown(); }

std::expected<void, GpuError> MetalBackend::initialize(const GpuConfig& /*config*/) {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices.count == 0) {
            return std::unexpected(GpuError::DeviceNotFound);
        }

        impl_->device = devices[0];
        impl_->queue = [impl_->device newCommandQueue];
        impl_->has_unified_memory = impl_->device.hasUnifiedMemory;

        impl_->info.name = [impl_->device.name UTF8String];
        impl_->info.vendor = "Apple";
        impl_->info.backend = "metal";
        impl_->info.device_index = 0;
        impl_->info.has_unified_memory = impl_->device.hasUnifiedMemory;
        impl_->info.total_device_memory_bytes = impl_->device.recommendedMaxWorkingSetSize;
        impl_->info.free_device_memory_bytes = impl_->info.total_device_memory_bytes / 2;
        impl_->info.supports_fp16 = [impl_->device supportsFamily:MTLGPUFamilyApple7];
        impl_->info.supports_bf16 = [impl_->device supportsFamily:MTLGPUFamilyApple9];
        impl_->info.supports_int8 = true;

        impl_->info.max_threads_per_block =
            static_cast<uint32_t>(impl_->device.maxThreadsPerThreadgroup.width);
        impl_->info.multiprocessor_count = 0;
        impl_->info.shared_memory_per_block_bytes =
            static_cast<size_t>(impl_->device.maxThreadgroupMemoryLength);

        impl_->info.supports_cagra = false;
        impl_->info.supports_ivf_pq = true;
        impl_->info.supports_dynamic_batching = true;
        impl_->info.supports_half_precision_search = impl_->info.supports_fp16;

        NSString* libPath = findMetalLibPath();
        if (!libPath) {
            NSLog(@"[elips] elips_kernels.metallib not found. Set ELIPS_METALLIB_PATH "
                   "or rebuild; expected at %s", ELIPS_METALLIB_BUILD_PATH);
            return std::unexpected(GpuError::InitializationFailed);
        }
        NSError* error = nil;
        id<MTLLibrary> lib =
            [impl_->device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
        if (!lib) {
            NSLog(@"[elips] failed to load metallib at %@: %@", libPath, error);
            return std::unexpected(GpuError::InitializationFailed);
        }
        impl_->library = lib;
        impl_->initialized = true;
        return {};
    }
}

void MetalBackend::shutdown() noexcept {
    impl_->pipelines.clear();
    impl_->library = nil;
    impl_->queue = nil;
    impl_->device = nil;
    impl_->initialized = false;
}

GpuDeviceInfo MetalBackend::device_info() const noexcept { return impl_->info; }
bool MetalBackend::is_available() const noexcept { return impl_->initialized; }

std::expected<GpuBuffer, GpuError> MetalBackend::allocate_device(size_t bytes) {
    @autoreleasepool {
        MTLResourceOptions options = impl_->has_unified_memory
            ? MTLResourceStorageModeShared
            : MTLResourceStorageModePrivate;
        id<MTLBuffer> buf = [impl_->device newBufferWithLength:bytes options:options];
        if (!buf) return std::unexpected(GpuError::InsufficientMemory);
        return GpuBuffer{buf.contents, bytes, (__bridge_retained void*)buf};
    }
}

void MetalBackend::free_device(GpuBuffer buf) noexcept {
    if (buf.backend_handle()) {
        CFRelease(buf.backend_handle());
    }
}

std::expected<void, GpuError>
MetalBackend::upload(const void* host_src, GpuBuffer& dst, size_t bytes) {
    if (impl_->has_unified_memory) {
        std::memcpy(dst.device_ptr(), host_src, bytes);
    } else {
        id<MTLBuffer> tmp = [impl_->device newBufferWithBytes:host_src length:bytes
                                                       options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        id<MTLBuffer> dstBuf = (__bridge id<MTLBuffer>)dst.backend_handle();
        [blit copyFromBuffer:tmp sourceOffset:0 toBuffer:dstBuf destinationOffset:0 size:bytes];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    return {};
}

std::expected<void, GpuError>
MetalBackend::download(const GpuBuffer& src, void* host_dst, size_t bytes) {
    if (impl_->has_unified_memory) {
        std::memcpy(host_dst, src.device_ptr(), bytes);
    } else {
        id<MTLBuffer> tmp = [impl_->device newBufferWithLength:bytes
                                                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        id<MTLBuffer> srcBuf = (__bridge id<MTLBuffer>)src.backend_handle();
        [blit copyFromBuffer:srcBuf sourceOffset:0 toBuffer:tmp destinationOffset:0 size:bytes];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        std::memcpy(host_dst, tmp.contents, bytes);
    }
    return {};
}
// APPEND_MARKER
std::expected<void, GpuError>
MetalBackend::compute_distances_batch(
    std::span<const float> queries, std::span<const float> database,
    std::span<float> distances_out, size_t nq, size_t nb, size_t dim,
    elips::Metric metric) {
    if (nq == 0 || nb == 0) return {};
    @autoreleasepool {
        const char* kname = nullptr;
        uint32_t mode = 0;  // ip_matrix: 0 = cosine (1-dot), 1 = dot_product (-dot)
        switch (metric) {
            case elips::Metric::euclidean:   kname = "elips_l2_matrix"; break;
            case elips::Metric::cosine:      kname = "elips_ip_matrix"; mode = 0; break;
            case elips::Metric::dot_product: kname = "elips_ip_matrix"; mode = 1; break;
        }
        id<MTLComputePipelineState> pso = impl_->pipeline(kname);
        if (!pso) return std::unexpected(GpuError::KernelLaunchFailed);

        id<MTLBuffer> qBuf = [impl_->device newBufferWithBytes:queries.data()
                                                        length:queries.size_bytes()
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> dbBuf = [impl_->device newBufferWithBytes:database.data()
                                                         length:database.size_bytes()
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> distBuf = [impl_->device newBufferWithLength:distances_out.size_bytes()
                                                           options:MTLResourceStorageModeShared];
        uint32_t params[4] = {static_cast<uint32_t>(nq), static_cast<uint32_t>(nb),
                              static_cast<uint32_t>(dim), mode};

        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:qBuf offset:0 atIndex:0];
        [enc setBuffer:dbBuf offset:0 atIndex:1];
        [enc setBuffer:distBuf offset:0 atIndex:2];
        [enc setBytes:params length:sizeof(params) atIndex:3];

        // 32x32 output tile, 16x16 threads per threadgroup (each owns a 2x2 micro-tile).
        constexpr NSUInteger kTile = 32;
        MTLSize grid = MTLSizeMake((nb + kTile - 1) / kTile, (nq + kTile - 1) / kTile, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(distances_out.data(), distBuf.contents, distances_out.size_bytes());
        return {};
    }
}

namespace {
// Threadgroup width for topk_select: P = max(32, next_pow2(k)), capped at MAXP
// (1024). P >= k guarantees the exact top-k fits in the running best buffer.
NSUInteger topk_width(size_t k) {
    size_t p = 32;
    while (p < k) p <<= 1;
    return static_cast<NSUInteger>(p > 1024 ? 1024 : p);
}
}  // namespace

std::expected<void, GpuError>
MetalBackend::top_k(std::span<const float> distances, std::span<uint32_t> indices_out,
                    std::span<float> values_out, size_t nq, size_t nb, size_t k) {
    if (nq == 0 || k == 0) return {};
    @autoreleasepool {
        id<MTLComputePipelineState> pso = impl_->pipeline("topk_select");
        if (!pso) return std::unexpected(GpuError::KernelLaunchFailed);
        NSUInteger P = topk_width(k);

        id<MTLBuffer> distBuf = [impl_->device newBufferWithBytes:distances.data()
                                                           length:distances.size_bytes()
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> outDist = [impl_->device newBufferWithLength:nq * k * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> outIdx = [impl_->device newBufferWithLength:nq * k * sizeof(int32_t)
                                                          options:MTLResourceStorageModeShared];
        // ELIPS distances are ordering-normalized (smaller == better) for every
        // metric, so top-k always keeps the smallest.
        uint32_t params[4] = {static_cast<uint32_t>(nq), static_cast<uint32_t>(nb),
                              static_cast<uint32_t>(k), static_cast<uint32_t>(P)};

        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:distBuf offset:0 atIndex:0];
        [enc setBuffer:outDist offset:0 atIndex:1];
        [enc setBuffer:outIdx offset:0 atIndex:2];
        [enc setBytes:params length:sizeof(params) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(nq, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(P, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(values_out.data(), outDist.contents, nq * k * sizeof(float));
        // Kernel writes int32 (-1 pads for k > nb); reinterpret into uint32 out.
        std::memcpy(indices_out.data(), outIdx.contents, nq * k * sizeof(int32_t));
        return {};
    }
}

std::expected<void, GpuError>
MetalBackend::ivf_scan(std::span<const float> query, std::span<const float> candidate_vectors,
                       std::span<const uint32_t> list_offsets, std::span<uint32_t> indices_out,
                       std::span<float> values_out, size_t nlists, size_t ncand, size_t dim,
                       size_t k, elips::Metric metric) {
    if (k == 0 || nlists == 0 || ncand == 0) return {};
    @autoreleasepool {
        id<MTLComputePipelineState> scanPso = impl_->pipeline("elips_ivf_scan_list");
        id<MTLComputePipelineState> mergePso = impl_->pipeline("elips_ivf_merge_lists");
        if (!scanPso || !mergePso) return std::unexpected(GpuError::KernelLaunchFailed);

        uint32_t mode = 0;  // 0 euclidean (sqrt L2), 1 cosine (1-dot), 2 dot (-dot)
        switch (metric) {
            case elips::Metric::euclidean:   mode = 0; break;
            case elips::Metric::cosine:      mode = 1; break;
            case elips::Metric::dot_product: mode = 2; break;
        }
        const NSUInteger P = topk_width(k);

        id<MTLBuffer> qBuf = [impl_->device newBufferWithBytes:query.data()
                                                        length:query.size_bytes()
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> vBuf = [impl_->device newBufferWithBytes:candidate_vectors.data()
                                                        length:candidate_vectors.size_bytes()
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> offBuf = [impl_->device newBufferWithBytes:list_offsets.data()
                                                          length:list_offsets.size_bytes()
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> perDist = [impl_->device newBufferWithLength:nlists * k * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> perIdx = [impl_->device newBufferWithLength:nlists * k * sizeof(int32_t)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> outDist = [impl_->device newBufferWithLength:k * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> outIdx = [impl_->device newBufferWithLength:k * sizeof(int32_t)
                                                          options:MTLResourceStorageModeShared];

        uint32_t scanParams[5] = {static_cast<uint32_t>(nlists), static_cast<uint32_t>(dim),
                                  static_cast<uint32_t>(k), static_cast<uint32_t>(P), mode};
        uint32_t mergeParams[3] = {static_cast<uint32_t>(nlists * k),
                                   static_cast<uint32_t>(k), static_cast<uint32_t>(P)};

        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        // Pass 1: one threadgroup per probed list -> per-list top-k.
        id<MTLComputeCommandEncoder> scan = [cmd computeCommandEncoder];
        [scan setComputePipelineState:scanPso];
        [scan setBuffer:qBuf offset:0 atIndex:0];
        [scan setBuffer:vBuf offset:0 atIndex:1];
        [scan setBuffer:offBuf offset:0 atIndex:2];
        [scan setBuffer:perDist offset:0 atIndex:3];
        [scan setBuffer:perIdx offset:0 atIndex:4];
        [scan setBytes:scanParams length:sizeof(scanParams) atIndex:5];
        [scan dispatchThreadgroups:MTLSizeMake(nlists, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(P, 1, 1)];
        [scan endEncoding];

        // Pass 2: merge nlists*k per-list candidates -> final top-k.
        id<MTLComputeCommandEncoder> merge = [cmd computeCommandEncoder];
        [merge setComputePipelineState:mergePso];
        [merge setBuffer:perDist offset:0 atIndex:0];
        [merge setBuffer:perIdx offset:0 atIndex:1];
        [merge setBuffer:outDist offset:0 atIndex:2];
        [merge setBuffer:outIdx offset:0 atIndex:3];
        [merge setBytes:mergeParams length:sizeof(mergeParams) atIndex:4];
        [merge dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(P, 1, 1)];
        [merge endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(values_out.data(), outDist.contents, k * sizeof(float));
        std::memcpy(indices_out.data(), outIdx.contents, k * sizeof(int32_t));
        return {};
    }
}

void MetalBackend::synchronize() {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

bool MetalBackend::is_idle() const noexcept { return true; }

} // namespace elips::gpu::metal

#endif // ELIPS_METAL_ENABLED
