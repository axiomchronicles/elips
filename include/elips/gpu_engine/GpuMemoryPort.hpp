#ifndef ELIPS_GPU_ENGINE_GPU_MEMORY_PORT_HPP
#define ELIPS_GPU_ENGINE_GPU_MEMORY_PORT_HPP

#include <cstddef>

#include "elips/domain/Expected.hpp"
#include "elips/gpu_engine/GpuBuffer.hpp"
#include "elips/gpu_engine/GpuPort.hpp"

namespace elips::gpu {

class GpuMemoryPort {
public:
    virtual ~GpuMemoryPort() = default;

    [[nodiscard]] virtual std::expected<GpuBuffer, GpuError>
    allocate(size_t bytes, size_t alignment = 256) = 0;

    virtual void deallocate(GpuBuffer&& buf) noexcept = 0;

    [[nodiscard]] virtual std::expected<void*, GpuError>
    allocate_pinned(size_t bytes) = 0;

    virtual void deallocate_pinned(void* ptr) noexcept = 0;

    [[nodiscard]] virtual size_t bytes_used() const noexcept = 0;
    [[nodiscard]] virtual size_t bytes_available() const noexcept = 0;
    [[nodiscard]] virtual size_t peak_bytes_used() const noexcept = 0;
};

} // namespace elips::gpu

#endif