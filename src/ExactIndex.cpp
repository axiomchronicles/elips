#include "elips/index_engine/ExactIndex.hpp"

#include <algorithm>
#include <expected>
#include <string>
#include <string_view>

#include "elips/domain/Errors.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips {
namespace {

void validate_dimension(std::span<const float> vector, std::uint16_t dimension,
                        std::string_view label) {
    if (vector.size() != dimension) {
        throw DimensionMismatch{std::string(label) +
                                " dimension does not match index"};
    }
}

}  // namespace

void ExactIndex::insert(const RecordID& id, std::span<const float> vector) {
    validate_dimension(vector, dimension_, "vector");
    ids_.push_back(id);
    if (compressed()) {
        const std::size_t offset = codes_.size();
        codes_.resize(offset + row_width_);
        quantizer_->encode(
            vector, std::span<std::uint8_t>{codes_}.subspan(offset, row_width_));
        return;
    }
    data_.insert(data_.end(), vector.begin(), vector.end());
}

void ExactIndex::remove(const RecordID& id) {
    const auto it = std::find(ids_.begin(), ids_.end(), id);
    if (it == ids_.end()) {
        return;
    }
    const auto row = static_cast<std::size_t>(it - ids_.begin());
    ids_.erase(it);

    if (compressed()) {
        const auto first =
            codes_.begin() + static_cast<std::ptrdiff_t>(row * row_width_);
        codes_.erase(first, first + static_cast<std::ptrdiff_t>(row_width_));
        return;
    }
    const auto first = data_.begin() + static_cast<std::ptrdiff_t>(row * dimension_);
    data_.erase(first, first + dimension_);
}

std::vector<IndexPort::Hit> ExactIndex::search(std::span<const float> query,
                                               std::size_t k) const {
    validate_dimension(query, dimension_, "query");

    std::vector<Hit> scored;
    scored.reserve(ids_.size());

    if (compressed()) {
        // One table build for the whole scan, then a gather per row. This is
        // what makes a compressed scan cheaper than the fp32 one it replaces
        // rather than merely smaller.
        const auto lut = quantizer_->make_lut(query);
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            const std::span<const std::uint8_t> code{
                codes_.data() + (i * row_width_), row_width_};
            scored.emplace_back(ids_[i], quantizer_->lut_distance(lut, code));
        }
    } else {
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            const std::span<const float> row{data_.data() + (i * dimension_),
                                             dimension_};
            scored.emplace_back(ids_[i], distance(metric_, query, row));
        }
    }

    const std::size_t take = std::min(k, scored.size());
    std::partial_sort(
        scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(take),
        scored.end(),
        [](const Hit& lhs, const Hit& rhs) { return lhs.second < rhs.second; });
    scored.resize(take);
    return scored;
}

std::expected<IndexSnapshot, std::string> ExactIndex::export_snapshot() const {
    IndexSnapshot snapshot;
    snapshot.kind = IndexSnapshotKind::exact;
    snapshot.metric = metric_;
    snapshot.dimension = dimension_;
    snapshot.ids = ids_;

    if (!compressed()) {
        snapshot.vectors = data_;
        return snapshot;
    }

    // A snapshot's `vectors` field is the cross-backend interchange format and
    // is always fp32, so a compressed index decodes on the way out. The codes
    // ride along in `pq` for backends that can consume them directly.
    snapshot.vectors.resize(ids_.size() * static_cast<std::size_t>(dimension_));
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        quantizer_->decode(
            std::span<const std::uint8_t>{codes_.data() + (i * row_width_),
                                          row_width_},
            std::span<float>{snapshot.vectors}.subspan(i * dimension_, dimension_));
    }

    PqSnapshot pq;
    pq.codec = quantizer_->codec();
    pq.pq_dim = static_cast<std::uint32_t>(quantizer_->code_bytes());
    pq.codes = codes_;
    snapshot.pq = std::move(pq);
    return snapshot;
}

std::expected<void, std::string>
ExactIndex::import_snapshot(const IndexSnapshot& snapshot) {
    if (snapshot.dimension != dimension_) {
        return std::unexpected("snapshot dimension does not match ExactIndex");
    }
    if (snapshot.metric != metric_) {
        return std::unexpected("snapshot metric does not match ExactIndex");
    }
    if (snapshot.vectors.size() !=
        snapshot.ids.size() * static_cast<std::size_t>(dimension_)) {
        return std::unexpected("snapshot vector payload is not row-major exact data");
    }

    ids_ = snapshot.ids;
    if (!compressed()) {
        data_ = snapshot.vectors;
        codes_.clear();
        return {};
    }

    // Re-encode from the fp32 payload rather than trusting snapshot.pq: those
    // codes were produced by whatever quantizer wrote the snapshot, which need
    // not be the one attached here.
    data_.clear();
    codes_.assign(ids_.size() * row_width_, 0U);
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        quantizer_->encode(
            std::span<const float>{snapshot.vectors.data() + (i * dimension_),
                                   dimension_},
            std::span<std::uint8_t>{codes_}.subspan(i * row_width_, row_width_));
    }
    return {};
}

}  // namespace elips
