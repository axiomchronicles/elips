#ifndef ELIPS_GPU_ENGINE_GPU_INDEX_TRANSFER_MANAGER_HPP
#define ELIPS_GPU_ENGINE_GPU_INDEX_TRANSFER_MANAGER_HPP

#include <expected>
#include <string>

#include "elips/gpu_engine/GpuIndexPort.hpp"
#include "elips/index_engine/IndexTransferPort.hpp"

namespace elips::gpu {

class GpuIndexTransferManager {
public:
    [[nodiscard]] static std::expected<void, std::string>
    clone(const elips::IndexTransferPort& source,
          elips::IndexTransferPort& destination);

    [[nodiscard]] static std::expected<void, GpuError>
    clone_cpu_to_gpu(const elips::IndexTransferPort& source,
                     GpuIndexPort& destination);

    [[nodiscard]] static std::expected<void, std::string>
    clone_gpu_to_cpu(const GpuIndexPort& source,
                     elips::IndexTransferPort& destination);
};

}  // namespace elips::gpu

#endif  // ELIPS_GPU_ENGINE_GPU_INDEX_TRANSFER_MANAGER_HPP
