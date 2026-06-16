#ifndef ELIPS_INDEX_ENGINE_INDEX_TRANSFER_PORT_HPP
#define ELIPS_INDEX_ENGINE_INDEX_TRANSFER_PORT_HPP

#include <expected>
#include <string>

#include "elips/index_engine/IndexSnapshot.hpp"

namespace elips {

class IndexTransferPort {
public:
    virtual ~IndexTransferPort() = default;

    [[nodiscard]] virtual std::expected<IndexSnapshot, std::string>
    export_snapshot() const = 0;

    [[nodiscard]] virtual std::expected<void, std::string>
    import_snapshot(const IndexSnapshot& snapshot) = 0;
};

}  // namespace elips

#endif  // ELIPS_INDEX_ENGINE_INDEX_TRANSFER_PORT_HPP
