#include "IvfIndexState.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string_view>

#include "elips/domain/Errors.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu::detail {
namespace {

std::size_t normalize_target_lists(std::size_t requested,
                                   std::size_t available) noexcept {
    return std::max<std::size_t>(1, std::min(requested, available));
}

void normalize_if_needed(elips::Metric metric, std::span<float> row) {
    if (!elips::requires_normalization(metric)) {
        return;
    }
    float norm = 0.0F;
    for (float value : row) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm <= 0.0F) {
        return;
    }
    for (float& value : row) {
        value /= norm;
    }
}

}  // namespace

IvfIndexState::IvfIndexState(elips::Metric metric, std::uint16_t dimension,
                             const GpuConfig& config)
    : metric_(metric),
      dimension_(dimension),
      config_(config),
      requested_n_lists_(std::max<std::size_t>(1, config.ivf_pq_params.n_lists)),
      nprobe_(std::max<std::size_t>(1, config.default_ef_search_gpu)) {}

IvfIndexState::~IvfIndexState() = default;

void IvfIndexState::clear() noexcept {
    ids_.clear();
    host_vectors_.clear();
    centroids_.clear();
    vector_to_list_.clear();
    list_members_.clear();
    id_to_slot_.clear();
    active_n_lists_ = 0;
    trained_ = false;
}

void IvfIndexState::validate_dimension(std::span<const float> vector,
                                       std::string_view label) const {
    if (vector.size() != dimension_) {
        throw elips::DimensionMismatch{std::string(label) +
                                       " dimension does not match IVF index"};
    }
}

void IvfIndexState::rebuild_id_map() {
    id_to_slot_.clear();
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        id_to_slot_[ids_[i]] = i;
    }
}

void IvfIndexState::rebuild_list_members() {
    list_members_.assign(active_n_lists_, {});
    for (std::uint32_t slot = 0; slot < vector_to_list_.size(); ++slot) {
        const auto list_id = vector_to_list_[slot];
        if (list_id < list_members_.size()) {
            list_members_[list_id].push_back(slot);
        }
    }
}

void IvfIndexState::sync_device_buffers_best_effort(GpuPort& backend) noexcept {
    if (vector_buffer_) {
        backend.free_device(std::move(vector_buffer_));
    }
    if (centroid_buffer_) {
        backend.free_device(std::move(centroid_buffer_));
    }

    if (!host_vectors_.empty()) {
        auto alloc = backend.allocate_device(host_vectors_.size() * sizeof(float));
        if (alloc.has_value()) {
            vector_buffer_ = std::move(*alloc);
            auto upload = backend.upload(host_vectors_.data(), vector_buffer_,
                                         host_vectors_.size() * sizeof(float));
            if (!upload.has_value()) {
                backend.free_device(std::move(vector_buffer_));
            }
        }
    }

    if (!centroids_.empty()) {
        auto alloc = backend.allocate_device(centroids_.size() * sizeof(float));
        if (alloc.has_value()) {
            centroid_buffer_ = std::move(*alloc);
            auto upload = backend.upload(centroids_.data(), centroid_buffer_,
                                         centroids_.size() * sizeof(float));
            if (!upload.has_value()) {
                backend.free_device(std::move(centroid_buffer_));
            }
        }
    }
}

std::vector<float> IvfIndexState::train_centroids(
    std::span<const float> vectors,
    std::size_t nvecs,
    std::size_t n_lists) const {
    std::vector<float> centroids(n_lists * dimension_);
    const std::size_t stride = std::max<std::size_t>(1, nvecs / n_lists);
    for (std::size_t list = 0; list < n_lists; ++list) {
        const auto sample = std::min(nvecs - 1, list * stride);
        std::copy_n(vectors.data() + sample * dimension_, dimension_,
                    centroids.data() + list * dimension_);
    }

    std::vector<std::uint32_t> assignments(nvecs, 0);
    const auto iterations =
        std::max<std::size_t>(1, config_.ivf_pq_params.kmeans_n_iters);
    for (std::size_t iter = 0; iter < iterations; ++iter) {
        assignments = assign_vectors_to_centroids(vectors, nvecs, centroids, n_lists);

        std::vector<float> sums(n_lists * dimension_, 0.0F);
        std::vector<std::size_t> counts(n_lists, 0);
        for (std::size_t vec = 0; vec < nvecs; ++vec) {
            const auto list = assignments[vec];
            ++counts[list];
            const auto* row = vectors.data() + vec * dimension_;
            auto* sum = sums.data() + list * dimension_;
            for (std::size_t dim = 0; dim < dimension_; ++dim) {
                sum[dim] += row[dim];
            }
        }

        for (std::size_t list = 0; list < n_lists; ++list) {
            auto* centroid = centroids.data() + list * dimension_;
            if (counts[list] == 0) {
                const auto fallback = std::min(nvecs - 1, list % nvecs);
                std::copy_n(vectors.data() + fallback * dimension_, dimension_, centroid);
            } else {
                for (std::size_t dim = 0; dim < dimension_; ++dim) {
                    centroid[dim] =
                        sums[list * dimension_ + dim] /
                        static_cast<float>(counts[list]);
                }
            }
            normalize_if_needed(metric_, std::span<float>{centroid, dimension_});
        }
    }

    return centroids;
}

std::vector<std::uint32_t> IvfIndexState::assign_vectors_to_centroids(
    std::span<const float> vectors,
    std::size_t nvecs,
    std::span<const float> centroids,
    std::size_t n_lists) const {
    std::vector<std::uint32_t> assignments(nvecs, 0);
    for (std::size_t vec = 0; vec < nvecs; ++vec) {
        const auto row = std::span<const float>{
            vectors.data() + vec * dimension_, static_cast<std::size_t>(dimension_)};
        float best_distance = std::numeric_limits<float>::max();
        std::uint32_t best_list = 0;
        for (std::uint32_t list = 0; list < n_lists; ++list) {
            const auto centroid = std::span<const float>{
                centroids.data() + list * dimension_,
                static_cast<std::size_t>(dimension_)};
            const auto current = elips::distance(metric_, row, centroid);
            if (current < best_distance) {
                best_distance = current;
                best_list = list;
            }
        }
        assignments[vec] = best_list;
    }
    return assignments;
}

void IvfIndexState::rebuild_from_current_data(GpuPort& backend, std::size_t n_lists) {
    if (ids_.empty()) {
        clear();
        sync_device_buffers_best_effort(backend);
        return;
    }

    active_n_lists_ = normalize_target_lists(n_lists, ids_.size());
    centroids_ = train_centroids(host_vectors_, ids_.size(), active_n_lists_);
    vector_to_list_ = assign_vectors_to_centroids(
        host_vectors_, ids_.size(), centroids_, active_n_lists_);
    rebuild_list_members();
    rebuild_id_map();
    trained_ = true;
    sync_device_buffers_best_effort(backend);
}

void IvfIndexState::maybe_retrain_after_growth(GpuPort& backend) {
    if (ids_.empty()) {
        clear();
        sync_device_buffers_best_effort(backend);
        return;
    }

    const auto target = normalize_target_lists(requested_n_lists_, ids_.size());
    if (!trained_) {
        rebuild_from_current_data(backend, target);
        return;
    }

    if (target > active_n_lists_ &&
        (target == requested_n_lists_ || ids_.size() >= active_n_lists_ * 2)) {
        rebuild_from_current_data(backend, target);
        return;
    }

    sync_device_buffers_best_effort(backend);
}

void IvfIndexState::insert(GpuPort& backend, const RecordID& id,
                           std::span<const float> vector) {
    validate_dimension(vector, "vector");
    if (id_to_slot_.contains(id)) {
        remove(backend, id);
    }

    ids_.push_back(id);
    host_vectors_.insert(host_vectors_.end(), vector.begin(), vector.end());
    id_to_slot_[id] = ids_.size() - 1;

    if (!trained_) {
        maybe_retrain_after_growth(backend);
        return;
    }

    const auto assignment = assign_vectors_to_centroids(
        std::span<const float>{vector.data(), vector.size()},
        1,
        centroids_,
        active_n_lists_);
    vector_to_list_.push_back(assignment.front());
    if (assignment.front() >= list_members_.size()) {
        list_members_.resize(assignment.front() + 1);
    }
    list_members_[assignment.front()].push_back(
        static_cast<std::uint32_t>(ids_.size() - 1));
    maybe_retrain_after_growth(backend);
}

void IvfIndexState::remove(GpuPort& backend, const RecordID& id) {
    const auto found = id_to_slot_.find(id);
    if (found == id_to_slot_.end()) {
        return;
    }

    const auto slot = found->second;
    const auto last = ids_.size() - 1;

    if (trained_ && slot < vector_to_list_.size()) {
        const auto list_id = vector_to_list_[slot];
        auto& members = list_members_[list_id];
        const auto member_it =
            std::find(members.begin(), members.end(), static_cast<std::uint32_t>(slot));
        if (member_it != members.end()) {
            members.erase(member_it);
        }
    }

    const auto slot_begin =
        static_cast<std::ptrdiff_t>(slot * static_cast<std::size_t>(dimension_));
    const auto last_begin =
        static_cast<std::ptrdiff_t>(last * static_cast<std::size_t>(dimension_));

    if (slot != last) {
        ids_[slot] = ids_[last];
        std::copy(host_vectors_.begin() + last_begin,
                  host_vectors_.begin() + last_begin + dimension_,
                  host_vectors_.begin() + slot_begin);
        id_to_slot_[ids_[slot]] = slot;

        if (trained_) {
            vector_to_list_[slot] = vector_to_list_[last];
            auto& moved_members = list_members_[vector_to_list_[slot]];
            const auto moved_it = std::find(
                moved_members.begin(), moved_members.end(),
                static_cast<std::uint32_t>(last));
            if (moved_it != moved_members.end()) {
                *moved_it = static_cast<std::uint32_t>(slot);
            }
        }
    }

    ids_.pop_back();
    host_vectors_.erase(host_vectors_.begin() + last_begin,
                        host_vectors_.begin() + last_begin + dimension_);
    id_to_slot_.erase(found);

    if (trained_ && !vector_to_list_.empty()) {
        vector_to_list_.pop_back();
    }

    if (ids_.empty()) {
        clear();
        sync_device_buffers_best_effort(backend);
        return;
    }

    if (trained_ && active_n_lists_ > ids_.size()) {
        rebuild_from_current_data(backend, ids_.size());
        return;
    }

    sync_device_buffers_best_effort(backend);
}

std::expected<void, std::string> IvfIndexState::build_from_batch(
    GpuPort& backend, std::span<const float> vectors, std::span<const RecordID> ids) {
    if (vectors.size() != ids.size() * static_cast<std::size_t>(dimension_)) {
        return std::unexpected("IVF batch build vectors are not row-major");
    }
    ids_.assign(ids.begin(), ids.end());
    host_vectors_.assign(vectors.begin(), vectors.end());
    rebuild_from_current_data(backend, normalize_target_lists(requested_n_lists_, ids_.size()));
    return {};
}

std::expected<void, std::string> IvfIndexState::import_snapshot(
    GpuPort& backend, const elips::IndexSnapshot& snapshot) {
    if (snapshot.dimension != dimension_) {
        return std::unexpected("snapshot dimension does not match IVF index");
    }
    if (snapshot.metric != metric_) {
        return std::unexpected("snapshot metric does not match IVF index");
    }
    if (snapshot.vectors.size() !=
        snapshot.ids.size() * static_cast<std::size_t>(dimension_)) {
        return std::unexpected("snapshot vector payload is not row-major IVF data");
    }

    ids_ = snapshot.ids;
    host_vectors_ = snapshot.vectors;
    rebuild_id_map();

    if (snapshot.ivf.has_value() &&
        snapshot.ivf->centroids.size() ==
            snapshot.ivf->n_lists * static_cast<std::size_t>(dimension_) &&
        snapshot.ivf->assignments.size() == ids_.size()) {
        active_n_lists_ = normalize_target_lists(snapshot.ivf->n_lists, ids_.size());
        nprobe_ = std::max<std::size_t>(1, snapshot.ivf->n_probe);
        centroids_ = snapshot.ivf->centroids;
        centroids_.resize(active_n_lists_ * static_cast<std::size_t>(dimension_));
        vector_to_list_ = snapshot.ivf->assignments;
        rebuild_list_members();
        trained_ = !ids_.empty();
        sync_device_buffers_best_effort(backend);
        return {};
    }

    rebuild_from_current_data(backend,
                              normalize_target_lists(requested_n_lists_, ids_.size()));
    return {};
}

elips::IndexSnapshot IvfIndexState::export_snapshot(
    elips::IndexSnapshotKind kind) const {
    elips::IndexSnapshot snapshot;
    snapshot.kind = kind;
    snapshot.metric = metric_;
    snapshot.dimension = dimension_;
    snapshot.ids = ids_;
    snapshot.vectors = host_vectors_;
    if (trained_ && active_n_lists_ > 0) {
        elips::IvfSnapshot ivf;
        ivf.n_lists = active_n_lists_;
        ivf.n_probe = nprobe_;
        ivf.centroids = centroids_;
        ivf.assignments = vector_to_list_;
        snapshot.ivf = std::move(ivf);
    }
    return snapshot;
}

std::expected<std::vector<std::uint32_t>, GpuError> IvfIndexState::probe_lists(
    GpuPort& backend, std::span<const float> query, std::size_t nprobe) const {
    validate_dimension(query, "query");
    if (!trained_ || centroids_.empty()) {
        return std::vector<std::uint32_t>{};
    }

    const auto take = std::min(nprobe, active_n_lists_);
    std::vector<float> distances(active_n_lists_);
    auto dist = backend.compute_distances_batch(
        query,
        std::span<const float>{centroids_.data(), centroids_.size()},
        distances,
        1,
        active_n_lists_,
        dimension_,
        metric_);
    if (!dist.has_value()) {
        return std::unexpected(dist.error());
    }

    std::vector<std::uint32_t> indices(take);
    std::vector<float> values(take);
    auto tk = backend.top_k(distances, indices, values, 1, active_n_lists_, take);
    if (!tk.has_value()) {
        return std::unexpected(tk.error());
    }
    return indices;
}

std::expected<std::vector<std::vector<std::uint32_t>>, GpuError>
IvfIndexState::probe_lists_batch(GpuPort& backend, std::span<const float> queries,
                                 std::size_t nprobe) const {
    if (queries.size() % dimension_ != 0) {
        return std::unexpected(GpuError::IndexBuildFailed);
    }
    const auto nq = queries.size() / dimension_;
    if (!trained_ || centroids_.empty()) {
        return std::vector<std::vector<std::uint32_t>>(nq);
    }

    const auto take = std::min(nprobe, active_n_lists_);
    std::vector<float> distances(nq * active_n_lists_);
    auto dist = backend.compute_distances_batch(
        queries,
        std::span<const float>{centroids_.data(), centroids_.size()},
        distances,
        nq,
        active_n_lists_,
        dimension_,
        metric_);
    if (!dist.has_value()) {
        return std::unexpected(dist.error());
    }

    std::vector<std::uint32_t> indices(nq * take);
    std::vector<float> values(nq * take);
    auto tk = backend.top_k(distances, indices, values, nq, active_n_lists_, take);
    if (!tk.has_value()) {
        return std::unexpected(tk.error());
    }

    std::vector<std::vector<std::uint32_t>> out(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        auto& row = out[q];
        row.assign(indices.begin() + static_cast<std::ptrdiff_t>(q * take),
                   indices.begin() + static_cast<std::ptrdiff_t>((q + 1) * take));
    }
    return out;
}

IvfIndexState::CandidateSet IvfIndexState::gather_candidates(
    const std::vector<std::uint32_t>& lists) const {
    CandidateSet candidates;
    if (!trained_) {
        return all_candidates();
    }

    std::size_t total = 0;
    for (auto list : lists) {
        if (list < list_members_.size()) {
            total += list_members_[list].size();
        }
    }

    candidates.ids.reserve(total);
    candidates.vectors.reserve(total * static_cast<std::size_t>(dimension_));
    candidates.source_slots.reserve(total);
    candidates.source_lists.reserve(total);

    for (auto list : lists) {
        if (list >= list_members_.size()) {
            continue;
        }
        for (auto slot : list_members_[list]) {
            candidates.ids.push_back(ids_[slot]);
            candidates.source_slots.push_back(slot);
            candidates.source_lists.push_back(list);
            const auto offset =
                static_cast<std::size_t>(slot) * static_cast<std::size_t>(dimension_);
            candidates.vectors.insert(
                candidates.vectors.end(),
                host_vectors_.begin() + static_cast<std::ptrdiff_t>(offset),
                host_vectors_.begin() +
                    static_cast<std::ptrdiff_t>(offset + dimension_));
        }
    }

    return candidates;
}

IvfIndexState::CandidateSet IvfIndexState::all_candidates() const {
    CandidateSet candidates;
    candidates.ids = ids_;
    candidates.vectors = host_vectors_;
    candidates.source_slots.resize(ids_.size());
    candidates.source_lists.resize(ids_.size(), 0);
    std::iota(candidates.source_slots.begin(), candidates.source_slots.end(), 0U);
    return candidates;
}

std::size_t IvfIndexState::device_bytes_used() const noexcept {
    return vector_buffer_.bytes() + centroid_buffer_.bytes();
}

}  // namespace elips::gpu::detail
