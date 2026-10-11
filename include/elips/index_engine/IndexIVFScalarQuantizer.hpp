#ifndef ELIPS_INDEX_ENGINE_INDEX_IVF_SCALAR_QUANTIZER_HPP
#define ELIPS_INDEX_ENGINE_INDEX_IVF_SCALAR_QUANTIZER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/Index.hpp"
#include "elips/quant_engine/ScalarQuantizer.hpp"

namespace elips {

// CPU inverted-file index with scalar-quantized postings (FAISS
// IndexIVFScalarQuantizer, QT_8bit). A coarse quantizer of `nlist` k-means
// centroids partitions the space; within each list a vector is stored as an SQ8
// code -- each component linearly mapped onto a byte against a per-dimension
// [min, max] range learned once over the whole training set.
//
// Residual handling: unlike IVFPQ, postings here encode the RAW vector, not the
// residual. SQ8's per-dimension min/max is already near-lossless (half a
// quantization step per component), so residual coding buys little, and a
// per-list residual range would mean one ScalarQuantizer per cell. A single
// global codec keeps the design flat and the LUT shareable across lists.
//
// search() probes the `nprobe` nearest lists and scores their codes with the
// ScalarQuantizer ADC/LUT (no per-candidate decode), then merges top-k. LOSSY
// but high-recall: a flat 4x compression (dimension bytes vs dimension*4).
class IndexIVFScalarQuantizer final : public Index {
public:
    IndexIVFScalarQuantizer(Metric metric, std::uint16_t dimension,
                            std::size_t nlist);

    // Lists probed per query; clamped to [1, nlist] at search time. Default 1.
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
        return "ivfsq";
    }

    [[nodiscard]] std::size_t nlist() const noexcept { return nlist_; }
    // Bytes a single posting occupies (dimension for SQ8), for reporting a
    // compression ratio against dimension * sizeof(float).
    [[nodiscard]] std::size_t code_bytes() const noexcept { return code_bytes_; }

private:
    [[nodiscard]] std::size_t assign(std::span<const float> vector) const noexcept;
    [[nodiscard]] std::span<const float> centroid(std::size_t list) const noexcept {
        return {centroids_.data() + (list * dimension_), dimension_};
    }

    struct InvList {
        std::vector<RecordID> ids;
        std::vector<std::uint8_t> codes;  // ids.size() * code_bytes_
    };

    Metric metric_;
    std::uint16_t dimension_;
    std::size_t nlist_;
    std::size_t code_bytes_;
    bool trained_{false};
    std::vector<float> centroids_;  // nlist_ * dimension_, row-major
    std::unique_ptr<quant::ScalarQuantizer> sq_;  // trained on raw vectors
    std::vector<InvList> lists_;
    std::size_t ntotal_{0};
};

}  // namespace elips

#endif  // ELIPS_INDEX_ENGINE_INDEX_IVF_SCALAR_QUANTIZER_HPP
