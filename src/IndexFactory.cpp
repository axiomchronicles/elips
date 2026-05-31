#include "elips/index_engine/IndexFactory.hpp"

#include "elips/index_engine/ExactIndex.hpp"
#include "elips/index_engine/HierarchicalGraphIndex.hpp"

namespace elips {

std::unique_ptr<IndexPort> make_index(const Config& config,
                                      std::uint16_t dimension) {
    switch (config.index()) {
        case IndexType::exact:
            return std::make_unique<ExactIndex>(config.metric(), dimension);
        case IndexType::graph:
            return std::make_unique<HierarchicalGraphIndex>(
                config.metric(), dimension, config.graph_params());
    }
    return std::make_unique<ExactIndex>(config.metric(), dimension);
}

}  // namespace elips
