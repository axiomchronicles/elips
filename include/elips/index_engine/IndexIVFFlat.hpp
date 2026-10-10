#ifndef ELIPS_INDEX_ENGINE_INDEX_IVF_FLAT_HPP
#define ELIPS_INDEX_ENGINE_INDEX_IVF_FLAT_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/Index.hpp"

namespace elips {

// CPU inverted-file index with full-precision (flat) postings. A coarse
// quantizer of `nlist` k-means centroids partitions the space; each vector is
// stored, uncompressed, in the inverted list of its nearest centroid. search()
// probes only the `nprobe` lists nearest the query and scans those with the
// exact Metric kernel, so cost scales with nprobe/nlist of the collection.
//
// Postings hold full fp32 vectors (no residual encoding -- residuals only buy
// anything for a quantized posting, i.e. IVFPQ, which is a later phase). As
// nprobe -> nlist the scan becomes exhaustive and results match IndexFlat
// exactly; nprobe is the recall/latency knob (FAISS default 1).
class IndexIVFFlat final : public Index {
public:
    IndexIVFFlat(Metric metric, std::uint16_t dimension, std::size_t nlist);

    // Lists probed per query. Public, mutable, FAISS-style; clamped to >=1 and
    // to nlist at search time. Default 1 (FAISS IndexIVF.h:69).
    std::size_t nprobe{1};

    void train(std::span<const float> training_vectors, std::size_t n) override;
    [[nodiscard]] bool is_trained() const noexcept override { return trained_; }

    void add_with_ids(const RecordID& id, std::span<const float> vector) override;
    void remove(const RecordID& id) override;
    [[nodiscard]] std::vector<Hit> search(std::span<const float> query,
                                          std::size_t k) const override;

    void reset() override;
    [[nodiscard]] std::size_t ntotal() const noexcept override { return ntotal_; }
    [[nodiscard]] std::string_view type_name() const noexcept override {
        return "ivfflat";
    }

    [[nodiscard]] std::size_t nlist() const noexcept { return nlist_; }

private:
    // Index of the centroid nearest `vector` under the configured metric.
    [[nodiscard]] std::size_t assign(std::span<const float> vector) const noexcept;
    [[nodiscard]] std::span<const float> centroid(std::size_t list) const noexcept {
        return {centroids_.data() + (list * dimension_), dimension_};
    }

    // One inverted list: parallel id / row-major fp32 arrays.
    struct InvList {
        std::vector<RecordID> ids;
        std::vector<float> data;
    };

    Metric metric_;
    std::uint16_t dimension_;
    std::size_t nlist_;
    bool trained_{false};
    std::vector<float> centroids_;   // nlist_ * dimension_, row-major
    std::vector<InvList> lists_;     // one per centroid
    std::size_t ntotal_{0};
};

}  // namespace elips

#endif  // ELIPS_INDEX_ENGINE_INDEX_IVF_FLAT_HPP
