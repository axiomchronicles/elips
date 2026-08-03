#ifndef ELIPS_GPU_ENGINE_GPU_KERNEL_PORT_HPP
#define ELIPS_GPU_ENGINE_GPU_KERNEL_PORT_HPP

#include <span>

#include "elips/Config.hpp"
#include "elips/domain/Expected.hpp"
#include "elips/gpu_engine/GpuPort.hpp"

namespace elips::gpu {

class GpuKernelPort {
public:
    virtual ~GpuKernelPort() = default;

    [[nodiscard]] virtual std::expected<void, GpuError>
    cosine_fp32(
        std::span<const float> queries,
        std::span<const float> database,
        std::span<float> distances,
        size_t nq, size_t nb, size_t dim) = 0;

    [[nodiscard]] virtual std::expected<void, GpuError>
    euclidean_fp32(
        std::span<const float> queries,
        std::span<const float> database,
        std::span<float> distances,
        size_t nq, size_t nb, size_t dim) = 0;

    [[nodiscard]] virtual std::expected<void, GpuError>
    dot_product_fp32(
        std::span<const float> queries,
        std::span<const float> database,
        std::span<float> distances,
        size_t nq, size_t nb, size_t dim) = 0;
};

} // namespace elips::gpu

#endif