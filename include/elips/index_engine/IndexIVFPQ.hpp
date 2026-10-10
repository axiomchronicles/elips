#ifndef ELIPS_INDEX_ENGINE_INDEX_IVF_PQ_HPP
#define ELIPS_INDEX_ENGINE_INDEX_IVF_PQ_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/Index.hpp"
#include "elips/quant_engine/ProductQuantizer.hpp"

namespace elips {

// CPU inverted-file index with product-quantized postings (FAISS IndexIVFPQ).
// A coarse quantizer of `nlist` k-means centroids partitions the space; within
// each list a vector is stored not raw but as a PQ code of its RESIDUAL -- the
// vector minus its assigned centroid (FAISS by_residual=true). Residual coding
// is the single biggest recall lever for IVFPQ: the residuals of one cell are
// tightly clustered near the origin, so a fixed PQ codebook represents them far
// better than it represents the raw, spread-out vectors.
//
// search() probes the `nprobe` nearest lists and scores their codes with the
// ProductQuantizer ADC/LUT (no per-candidate decode), then merges top-k. The
// index is LOSSY: recall rises monotonically with nprobe and never reaches the
// exact oracle, trading that for a pq_dim-byte footprint per vector instead of
// dimension*4.
class IndexIVFPQ final : public Index {
public:
    // pq_dim subspaces (FAISS M), pq_bits bits per subquantizer code (FAISS
    // nbits, default 8 -> one byte per subspace). dimension must be divisible
    // by pq_dim; pq_bits must be in [4, 8].
    IndexIVFPQ(Metric metric, std::uint16_t dimension, std::size_t nlist,
               std::uint32_t pq_dim, std::uint32_t pq_bits = 8);

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
        return "ivfpq";
    }

    [[nodiscard]] std::size_t nlist() const noexcept { return nlist_; }
    // Bytes a single posting occupies (pq_dim for 8-bit codes), for reporting a
    // compression ratio against dimension * sizeof(float).
    [[nodiscard]] std::size_t code_bytes() const noexcept { return code_bytes_; }

private:
    [[nodiscard]] std::size_t assign(std::span<const float> vector) const noexcept;
    [[nodiscard]] std::span<const float> centroid(std::size_t list) const noexcept {
        return {centroids_.data() + (list * dimension_), dimension_};
    }

    // One inverted list: parallel id / packed-code arrays. codes holds
    // ids.size() * code_bytes_ bytes, row-major.
    struct InvList {
        std::vector<RecordID> ids;
        std::vector<std::uint8_t> codes;
    };

    Metric metric_;
    std::uint16_t dimension_;
    std::size_t nlist_;
    std::uint32_t pq_dim_;
    std::uint32_t pq_bits_;
    std::size_t code_bytes_;
    bool trained_{false};
    std::vector<float> centroids_;  // nlist_ * dimension_, row-major
    std::unique_ptr<quant::ProductQuantizer> pq_;  // trained on residuals
    std::vector<InvList> lists_;
    std::size_t ntotal_{0};
};

}  // namespace elips

#endif  // ELIPS_INDEX_ENGINE_INDEX_IVF_PQ_HPP
