#ifndef ELIPS_INDEX_ENGINE_EXACT_INDEX_HPP
#define ELIPS_INDEX_ENGINE_EXACT_INDEX_HPP

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "elips/Config.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/index_engine/IndexPort.hpp"

namespace elips {

// Brute-force linear scan over (id, vector) pairs. Exact results; used for
// small collections and as the recall ground-truth oracle for ANN benchmarks.
class ExactIndex final : public IndexPort {
public:
    ExactIndex(Metric metric, std::uint16_t dimension) noexcept
        : metric_(metric), dimension_(dimension) {}

    void insert(const RecordID& id, std::span<const float> vector) override;
    void remove(const RecordID& id) override;
    [[nodiscard]] std::vector<Hit> search(std::span<const float> query,
                                          std::size_t k) const override;

    [[nodiscard]] std::size_t size() const noexcept override { return ids_.size(); }
    [[nodiscard]] std::string_view type_name() const noexcept override {
        return "exact";
    }

private:
    Metric metric_;
    std::uint16_t dimension_;
    std::vector<RecordID> ids_;
    std::vector<float> data_;  // row-major, dimension_ floats per record
};

}  // namespace elips

#endif  // ELIPS_INDEX_ENGINE_EXACT_INDEX_HPP
