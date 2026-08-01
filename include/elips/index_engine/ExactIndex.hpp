#ifndef ELIPS_INDEX_ENGINE_EXACT_INDEX_HPP
#define ELIPS_INDEX_ENGINE_EXACT_INDEX_HPP

#include <cstddef>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/IndexPort.hpp"
#include "elips/index_engine/IndexTransferPort.hpp"
#include "elips/quant_engine/Quantizer.hpp"

namespace elips {

// Brute-force linear scan over (id, vector) pairs. Exact results; used for
// small collections and as the recall ground-truth oracle for ANN benchmarks.
//
// When a trained quantizer is supplied, rows are stored as codes rather than
// fp32 and search computes distances by asymmetric lookup, which makes the scan
// both smaller and faster. Note that "exact" then refers to the scan being
// exhaustive, not to the distances being exact -- a quantized ExactIndex is no
// longer a ground-truth oracle.
class ExactIndex final : public IndexPort, public IndexTransferPort {
public:
    ExactIndex(Metric metric, std::uint16_t dimension,
               quant::QuantizerPtr quantizer = {}) noexcept
        : metric_(metric),
          dimension_(dimension),
          quantizer_(std::move(quantizer)),
          row_width_(quantizer_ != nullptr ? quantizer_->code_bytes()
                                           : dimension) {}

    void insert(const RecordID& id, std::span<const float> vector) override;
    void remove(const RecordID& id) override;
    [[nodiscard]] std::vector<Hit> search(std::span<const float> query,
                                          std::size_t k) const override;

    [[nodiscard]] std::size_t size() const noexcept override { return ids_.size(); }
    [[nodiscard]] std::string_view type_name() const noexcept override {
        return quantizer_ != nullptr ? "exact_quantized" : "exact";
    }

    [[nodiscard]] std::expected<IndexSnapshot, std::string>
    export_snapshot() const override;
    [[nodiscard]] std::expected<void, std::string>
    import_snapshot(const IndexSnapshot& snapshot) override;

private:
    // Row-major storage: dimension_ floats per record when uncompressed, or
    // code_bytes() bytes when a quantizer is attached. Only one is ever
    // populated.
    [[nodiscard]] bool compressed() const noexcept {
        return quantizer_ != nullptr;
    }

    Metric metric_;
    std::uint16_t dimension_;
    quant::QuantizerPtr quantizer_;
    std::size_t row_width_;
    std::vector<RecordID> ids_;
    std::vector<float> data_;           // used when uncompressed
    std::vector<std::uint8_t> codes_;   // used when compressed
};

}  // namespace elips

#endif  // ELIPS_INDEX_ENGINE_EXACT_INDEX_HPP
