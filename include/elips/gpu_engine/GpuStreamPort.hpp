#ifndef ELIPS_GPU_ENGINE_GPU_STREAM_PORT_HPP
#define ELIPS_GPU_ENGINE_GPU_STREAM_PORT_HPP

#include "elips/domain/Expected.hpp"
#include "elips/gpu_engine/GpuPort.hpp"

namespace elips::gpu {

class GpuStreamPort {
public:
    virtual ~GpuStreamPort() = default;

    [[nodiscard]] virtual std::expected<void, GpuError> synchronize() = 0;
    [[nodiscard]] virtual bool is_complete() const noexcept = 0;
    virtual void wait_for_completion() = 0;
};

} // namespace elips::gpu

#endif