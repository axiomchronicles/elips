// ELIPS minimal C API implementation. See include/elips/c_api/elips.h.
#include "elips/c_api/elips.h"

#include <array>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/Index.hpp"
#include "elips/index_engine/IndexFactory.hpp"

namespace {

// Encode an int64 id big-endian into the first 8 bytes of a RecordID; the rest
// stay zero. Reversible, so search can hand the original id back without a map.
elips::RecordID id_to_record(int64_t id) {
    elips::RecordID::Bytes bytes{};
    const auto u = static_cast<uint64_t>(id);
    for (int i = 0; i < 8; ++i) {
        bytes[static_cast<size_t>(i)] =
            static_cast<uint8_t>((u >> (56 - 8 * i)) & 0xFFU);
    }
    return elips::RecordID{bytes};
}

int64_t record_to_id(const elips::RecordID& rec) {
    const auto& bytes = rec.bytes();
    uint64_t u = 0;
    for (int i = 0; i < 8; ++i) {
        u = (u << 8) | bytes[static_cast<size_t>(i)];
    }
    return static_cast<int64_t>(u);
}

bool map_metric(elips_metric metric, elips::Metric& out) {
    switch (metric) {
        case ELIPS_METRIC_COSINE: out = elips::Metric::cosine; return true;
        case ELIPS_METRIC_EUCLIDEAN: out = elips::Metric::euclidean; return true;
        case ELIPS_METRIC_DOT_PRODUCT: out = elips::Metric::dot_product; return true;
    }
    return false;
}

}  // namespace

// The opaque handle just owns the C++ index.
struct elips_index {
    std::unique_ptr<elips::Index> impl;
    uint16_t dimension;
};

extern "C" {

elips_index* elips_index_create(uint16_t dimension, elips_metric metric,
                                const char* spec) {
    if (dimension == 0 || spec == nullptr) {
        return nullptr;
    }
    elips::Metric m{};
    if (!map_metric(metric, m)) {
        return nullptr;
    }
    try {
        auto impl = elips::index_factory(dimension, m, spec);
        return new elips_index{std::move(impl), dimension};
    } catch (...) {
        return nullptr;
    }
}

elips_status elips_index_add(elips_index* index, int64_t id,
                             const float* vector, uint16_t dimension) {
    if (index == nullptr || vector == nullptr || dimension != index->dimension) {
        return ELIPS_ERR_INVALID_ARG;
    }
    try {
        index->impl->add_with_ids(id_to_record(id),
                                  std::span<const float>(vector, dimension));
        return ELIPS_OK;
    } catch (...) {
        return ELIPS_ERR_RUNTIME;
    }
}

elips_status elips_index_search(const elips_index* index, const float* query,
                                uint16_t dimension, size_t k, int64_t* out_ids,
                                float* out_distances, size_t* out_count) {
    if (index == nullptr || query == nullptr || out_count == nullptr ||
        dimension != index->dimension) {
        return ELIPS_ERR_INVALID_ARG;
    }
    try {
        const auto hits = index->impl->search(
            std::span<const float>(query, dimension), k);
        const size_t n = hits.size() < k ? hits.size() : k;
        for (size_t i = 0; i < n; ++i) {
            if (out_ids != nullptr) {
                out_ids[i] = record_to_id(hits[i].first);
            }
            if (out_distances != nullptr) {
                out_distances[i] = hits[i].second;
            }
        }
        *out_count = n;
        return ELIPS_OK;
    } catch (...) {
        return ELIPS_ERR_RUNTIME;
    }
}

size_t elips_index_ntotal(const elips_index* index) {
    return index == nullptr ? 0 : index->impl->ntotal();
}

void elips_index_free(elips_index* index) { delete index; }

}  // extern "C"
