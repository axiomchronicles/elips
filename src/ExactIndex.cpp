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
    data_.insert(data_.end(), vector.begin(), vector.end());
}

void ExactIndex::remove(const RecordID& id) {
    const auto it = std::find(ids_.begin(), ids_.end(), id);
    if (it == ids_.end()) {
        return;
    }
    const auto row = static_cast<std::size_t>(it - ids_.begin());
    ids_.erase(it);
    const auto first = data_.begin() + static_cast<std::ptrdiff_t>(row * dimension_);
    data_.erase(first, first + dimension_);
}

std::vector<IndexPort::Hit> ExactIndex::search(std::span<const float> query,
                                               std::size_t k) const {
    validate_dimension(query, dimension_, "query");

    std::vector<Hit> scored;
    scored.reserve(ids_.size());
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        const std::span<const float> row{data_.data() + i * dimension_, dimension_};
        scored.emplace_back(ids_[i], distance(metric_, query, row));
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
    snapshot.vectors = data_;
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
    data_ = snapshot.vectors;
    return {};
}

}  // namespace elips
