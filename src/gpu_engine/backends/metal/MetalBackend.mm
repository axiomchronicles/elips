#ifdef ELIPS_METAL_ENABLED

#include "MetalBackend.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

#include <cstring>
#include <algorithm>
#include <vector>

namespace elips::gpu::metal {

struct MetalBackend::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    GpuDeviceInfo info;
    bool initialized{false};
    bool has_unified_memory{false};
};

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {
    impl_->device = nil;
    impl_->queue = nil;
    impl_->library = nil;
}

MetalBackend::~MetalBackend() { shutdown(); }

std::expected<void, GpuError> MetalBackend::initialize(const GpuConfig& config) {
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

        impl_->info.max_threads_per_block = 1024;
        impl_->info.multiprocessor_count = 0;
        impl_->info.shared_memory_per_block_bytes = 32768;

        impl_->info.supports_cagra = false;
        impl_->info.supports_ivf_pq = true;
        impl_->info.supports_dynamic_batching = true;
        impl_->info.supports_half_precision_search = impl_->info.supports_fp16;

        NSError* error = nil;
        NSString* shader_src = @""
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void cosine_fp32(\n"
            "    device const float* queries    [[buffer(0)]],\n"
            "    device const float* database   [[buffer(1)]],\n"
            "    device       float* distances  [[buffer(2)]],\n"
            "    constant uint& nq [[buffer(3)]],\n"
            "    constant uint& nb [[buffer(4)]],\n"
            "    constant uint& dim [[buffer(5)]],\n"
            "    uint2 gid [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (gid.x >= nq || gid.y >= nb) return;\n"
            "    float dot = 0.0, nq_mag = 0.0, nd_mag = 0.0;\n"
            "    for (uint d = 0; d < dim; ++d) {\n"
            "        float qv = queries[gid.x * dim + d];\n"
            "        float dv = database[gid.y * dim + d];\n"
            "        dot += qv * dv;\n"
            "        nq_mag += qv * qv;\n"
            "        nd_mag += dv * dv;\n"
            "    }\n"
            "    distances[gid.x * nb + gid.y] = 1.0 - dot / (sqrt(nq_mag) * sqrt(nd_mag) + 1e-8);\n"
            "}\n"
            "kernel void euclidean_fp32(\n"
            "    device const float* queries    [[buffer(0)]],\n"
            "    device const float* database   [[buffer(1)]],\n"
            "    device       float* distances  [[buffer(2)]],\n"
            "    constant uint& nq [[buffer(3)]],\n"
            "    constant uint& nb [[buffer(4)]],\n"
            "    constant uint& dim [[buffer(5)]],\n"
            "    uint2 gid [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (gid.x >= nq || gid.y >= nb) return;\n"
            "    float sum = 0.0;\n"
            "    for (uint d = 0; d < dim; ++d) {\n"
            "        float diff = queries[gid.x * dim + d] - database[gid.y * dim + d];\n"
            "        sum += diff * diff;\n"
            "    }\n"
            "    distances[gid.x * nb + gid.y] = sqrt(sum);\n"
            "}\n"
            "kernel void dot_product_fp32(\n"
            "    device const float* queries    [[buffer(0)]],\n"
            "    device const float* database   [[buffer(1)]],\n"
            "    device       float* distances  [[buffer(2)]],\n"
            "    constant uint& nq [[buffer(3)]],\n"
            "    constant uint& nb [[buffer(4)]],\n"
            "    constant uint& dim [[buffer(5)]],\n"
            "    uint2 gid [[thread_position_in_grid]]\n"
            ") {\n"
            "    if (gid.x >= nq || gid.y >= nb) return;\n"
            "    float sum = 0.0;\n"
            "    for (uint d = 0; d < dim; ++d) {\n"
            "        sum += queries[gid.x * dim + d] * database[gid.y * dim + d];\n"
            "    }\n"
            "    distances[gid.x * nb + gid.y] = -sum;\n"
            "}\n";

        id<MTLLibrary> lib = [impl_->device newLibraryWithSource:shader_src
                                                          options:nil error:&error];
        if (!lib) {
            return std::unexpected(GpuError::InitializationFailed);
        }
        impl_->library = lib;
        impl_->initialized = true;
        return {};
    }
}

void MetalBackend::shutdown() noexcept {
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

void MetalBackend::free_device(GpuBuffer&& buf) noexcept {
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

std::expected<void, GpuError>
MetalBackend::compute_distances_batch(
    std::span<const float> queries, std::span<const float> database,
    std::span<float> distances_out, size_t nq, size_t nb, size_t dim,
    elips::Metric metric) {
    @autoreleasepool {
        NSString* kernel_name;
        switch (metric) {
            case elips::Metric::cosine:      kernel_name = @"cosine_fp32"; break;
            case elips::Metric::euclidean:   kernel_name = @"euclidean_fp32"; break;
            case elips::Metric::dot_product: kernel_name = @"dot_product_fp32"; break;
        }

        id<MTLFunction> fn = [impl_->library newFunctionWithName:kernel_name];
        if (!fn) return std::unexpected(GpuError::KernelLaunchFailed);

        NSError* err = nil;
        id<MTLComputePipelineState> pso = [impl_->device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) return std::unexpected(GpuError::KernelLaunchFailed);

        id<MTLBuffer> qBuf = [impl_->device newBufferWithBytes:queries.data()
                                                         length:queries.size_bytes()
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> dbBuf = [impl_->device newBufferWithBytes:database.data()
                                                          length:database.size_bytes()
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> distBuf = [impl_->device newBufferWithLength:distances_out.size_bytes()
                                                            options:MTLResourceStorageModeShared];
        uint32_t nq32 = static_cast<uint32_t>(nq);
        uint32_t nb32 = static_cast<uint32_t>(nb);
        uint32_t dim32 = static_cast<uint32_t>(dim);

        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:qBuf offset:0 atIndex:0];
        [enc setBuffer:dbBuf offset:0 atIndex:1];
        [enc setBuffer:distBuf offset:0 atIndex:2];
        [enc setBytes:&nq32 length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&nb32 length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&dim32 length:sizeof(uint32_t) atIndex:5];

        MTLSize threads = MTLSizeMake(static_cast<NSUInteger>(nq), static_cast<NSUInteger>(nb), 1);
        MTLSize threadgroups = MTLSizeMake(1, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroups];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(distances_out.data(), distBuf.contents, distances_out.size_bytes());
        return {};
    }
}

std::expected<void, GpuError>
MetalBackend::top_k(std::span<const float> distances, std::span<uint32_t> indices_out,
                    std::span<float> values_out, size_t nq, size_t nb, size_t k) {
    for (size_t q = 0; q < nq; ++q) {
        std::vector<std::pair<float, uint32_t>> heap;
        heap.reserve(nb);
        for (size_t j = 0; j < nb; ++j) {
            heap.emplace_back(distances[q * nb + j], static_cast<uint32_t>(j));
        }
        std::partial_sort(heap.begin(), heap.begin() + static_cast<ptrdiff_t>(k),
                          heap.end());
        for (size_t j = 0; j < k; ++j) {
            indices_out[q * k + j] = heap[j].second;
            values_out[q * k + j] = heap[j].first;
        }
    }
    return {};
}

void MetalBackend::synchronize() {}
bool MetalBackend::is_idle() const noexcept { return true; }

} // namespace elips::gpu::metal

#endif // ELIPS_METAL_ENABLED