#include "elips/gpu_engine/GpuBruteForceIndex.hpp"

#include <algorithm>
#include <string>
#include <string_view>

#include "elips/gpu_engine/GpuSearchPipeline.hpp"
#include "elips/domain/Errors.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu {
namespace {

void validate_dimension(std::span<const float> vector, uint16_t dimension,
                        std::string_view label) {
    if (vector.size() != dimension) {
        throw elips::DimensionMismatch{std::string(label) +
                                       " dimension does not match GPU index"};
    }
}

}  // namespace

GpuBruteForceIndex::GpuBruteForceIndex(GpuPort& backend, elips::Metric metric,
                                       uint16_t dimension, const GpuConfig& config)
    : backend_(backend), metric_(metric), dimension_(dimension) {
    (void)config;
}

GpuBruteForceIndex::~GpuBruteForceIndex() { release_buffer(); }

void GpuBruteForceIndex::release_buffer() noexcept {
    if (!database_buffer_) {
        return;
    }
    backend_.free_device(std::move(database_buffer_));
}

void GpuBruteForceIndex::sync_device_buffer_best_effort() noexcept {
    release_buffer();
    if (host_vectors_.empty()) {
        return;
    }

    auto staged = create_device_buffer_from_host(host_vectors_);
    if (staged.has_value()) {
        database_buffer_ = std::move(*staged);
    }
}

std::expected<GpuBuffer, GpuError>
GpuBruteForceIndex::create_device_buffer_from_host(std::span<const float> vectors) {
    if (vectors.empty()) {
        return GpuBuffer{};
    }

    const size_t bytes = vectors.size() * sizeof(float);
    auto alloc = backend_.allocate_device(bytes);
    if (!alloc.has_value()) {
        return std::unexpected(alloc.error());
    }

    GpuBuffer buffer = std::move(*alloc);
    auto upload_result = backend_.upload(vectors.data(), buffer, bytes);
    if (!upload_result.has_value()) {
        backend_.free_device(std::move(buffer));
        return std::unexpected(upload_result.error());
    }
    return buffer;
}

void GpuBruteForceIndex::insert(const RecordID& id, std::span<const float> vector) {
    validate_dimension(vector, dimension_, "vector");
    ids_.push_back(id);
    host_vectors_.insert(host_vectors_.end(), vector.begin(), vector.end());
    count_ = ids_.size();
    sync_device_buffer_best_effort();
}

void GpuBruteForceIndex::remove(const RecordID& id) {
    auto it = std::find(ids_.begin(), ids_.end(), id);
    if (it == ids_.end()) {
        return;
    }

    const auto row = static_cast<size_t>(it - ids_.begin());
    ids_.erase(it);
    host_vectors_.erase(host_vectors_.begin() +
                            static_cast<ptrdiff_t>(row * dimension_),
                        host_vectors_.begin() +
                            static_cast<ptrdiff_t>((row + 1) * dimension_));
    count_ = ids_.size();
    sync_device_buffer_best_effort();
}

std::vector<IndexPort::Hit> GpuBruteForceIndex::search(std::span<const float> query,
                                                        std::size_t k) const {
    validate_dimension(query, dimension_, "query");
    if (count_ == 0 || k == 0 || host_vectors_.empty()) return {};

    const size_t take = std::min(k, count_);

    std::vector<float> distances(count_);
    auto result = backend_.compute_distances_batch(
        query, std::span<const float>{host_vectors_.data(), host_vectors_.size()},
        distances, 1, count_, dimension_, metric_);
    if (!result.has_value()) return {};

    std::vector<uint32_t> indices(take);
    std::vector<float> vals(take);
    auto tk = backend_.top_k(distances, indices, vals, 1, count_, take);
    if (!tk.has_value()) return {};

    std::vector<Hit> hits;
    hits.reserve(take);
    for (size_t i = 0; i < take && indices[i] < ids_.size(); ++i) {
        hits.emplace_back(ids_[indices[i]], vals[i]);
    }
    return hits;
}

std::size_t GpuBruteForceIndex::size() const noexcept { return count_; }
std::string_view GpuBruteForceIndex::type_name() const noexcept { return "gpu_brute_force"; }

std::expected<void, GpuError>
GpuBruteForceIndex::build_from_batch(std::span<const float> vectors,
                                     std::span<const RecordID> ids,
                                     const GpuIndexBuildParams& params) {
    (void)params;
    if (vectors.size() != ids.size() * static_cast<size_t>(dimension_)) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }

    auto staged = create_device_buffer_from_host(vectors);
    if (!staged.has_value()) {
        return std::unexpected(staged.error());
    }

    release_buffer();
    database_buffer_ = std::move(*staged);
    host_vectors_.assign(vectors.begin(), vectors.end());
    ids_.assign(ids.begin(), ids.end());
    count_ = ids_.size();
    return {};
}

std::expected<std::vector<std::vector<SearchResult>>, GpuError>
GpuBruteForceIndex::search_batch(std::span<const float> queries, size_t k,
                                  size_t ef_search) const {
    (void)ef_search;
    if (queries.size() % dimension_ != 0) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }

    size_t nq = queries.size() / dimension_;
    if (count_ == 0 || k == 0) {
        return std::vector<std::vector<SearchResult>>(nq);
    }

    const size_t take = std::min(k, count_);
    GpuSearchPipeline pipeline(const_cast<GpuPort&>(backend_));
    return pipeline.batch_search(queries, nq,
        std::span<const float>{host_vectors_.data(), host_vectors_.size()},
        ids_, count_, dimension_, take, metric_);
}

std::expected<void, GpuError>
GpuBruteForceIndex::export_to_cpu_index(elips::IndexPort& cpu_index_out) const {
    for (size_t i = 0; i < ids_.size(); ++i) {
        cpu_index_out.insert(
            ids_[i],
            std::span<const float>{host_vectors_.data() + i * dimension_,
                                   dimension_});
    }
    return {};
}

std::expected<void, GpuError>
GpuBruteForceIndex::import_from_cpu_index(const elips::IndexPort&) { return {}; }

size_t GpuBruteForceIndex::device_bytes_used() const noexcept {
    return database_buffer_.bytes();
}

std::string_view GpuBruteForceIndex::backend_name() const noexcept {
    static const std::string name = backend_.device_info().backend;
    return name;
}

} // namespace elips::gpu
