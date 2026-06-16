#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <vector>

#include "TestRecordingBackend.hpp"
#include "elips/Config.hpp"
#include "elips/gpu_engine/GpuBruteForceIndex.hpp"
#include "elips/gpu_engine/GpuDistributedIndex.hpp"
#include "elips/gpu_engine/GpuGraphIndex.hpp"
#include "elips/gpu_engine/GpuHybridIndex.hpp"
#include "elips/gpu_engine/GpuIVFFlatIndex.hpp"
#include "elips/gpu_engine/GpuIVFPQIndex.hpp"
#include "elips/index_engine/ExactIndex.hpp"

namespace elips::gpu {
namespace {

constexpr std::uint16_t kDimension = 4;

std::vector<float> sample_vectors() {
    return {
        0.0F, 0.0F, 0.0F, 0.0F,
        0.1F, 0.1F, 0.0F, 0.0F,
        5.0F, 5.0F, 5.0F, 5.0F,
        5.2F, 5.1F, 5.0F, 5.0F,
        -4.0F, -4.0F, -4.0F, -4.0F,
        -4.1F, -4.0F, -4.0F, -4.0F,
    };
}

std::vector<RecordID> sample_ids() {
    std::vector<RecordID> ids;
    ids.reserve(6);
    for (size_t i = 0; i < 6; ++i) {
        ids.push_back(RecordID::generate());
    }
    return ids;
}

GpuConfig make_ivf_config() {
    GpuConfig config;
    config.default_ef_search_gpu = 2;
    config.ivf_pq_params.n_lists = 3;
    config.ivf_pq_params.kmeans_n_iters = 6;
    config.ivf_pq_params.pq_dim = 2;
    config.ivf_pq_params.pq_bits = 4;
    return config;
}

GpuIndexBuildParams make_graph_build_params() {
    GpuIndexBuildParams params;
    params.params = GraphIndexBuildParams{
        .intermediate_graph_degree = 8,
        .graph_degree = 4,
        .build_algo = GraphIndexBuildParams::BuildAlgo::IvfPq,
        .nn_descent_iterations = 10,
        .compression_ratio = 0.0F,
    };
    return params;
}

void populate_exact_index(ExactIndex& index,
                          std::span<const RecordID> ids,
                          std::span<const float> vectors) {
    for (size_t row = 0; row < ids.size(); ++row) {
        index.insert(
            ids[row],
            std::span<const float>{
                vectors.data() + row * static_cast<size_t>(kDimension),
                static_cast<size_t>(kDimension)});
    }
}

std::vector<RecordID> hit_ids(const std::vector<IndexPort::Hit>& hits) {
    std::vector<RecordID> ids;
    ids.reserve(hits.size());
    for (const auto& [id, distance] : hits) {
        (void)distance;
        ids.push_back(id);
    }
    return ids;
}

std::unique_ptr<GpuIndexPort> make_brute_force_child(test::RecordingBackend& backend,
                                                     const GpuConfig& config) {
    return std::make_unique<GpuBruteForceIndex>(
        backend, Metric::euclidean, kDimension, config);
}

TEST(GpuIVFFlatIndexTest, snapshot_round_trip_preserves_search_results) {
    test::RecordingBackend backend;
    const auto ids = sample_ids();
    const auto vectors = sample_vectors();
    auto config = make_ivf_config();

    GpuIVFFlatIndex index(backend, Metric::euclidean, kDimension, config);
    auto built = index.build_from_batch(vectors, ids, GpuIndexBuildParams{});
    ASSERT_TRUE(built.has_value());

    const std::array<float, kDimension> query{0.0F, 0.0F, 0.0F, 0.0F};
    const auto before = index.search(query, 2);
    ASSERT_EQ(before.size(), 2U);
    EXPECT_EQ(before.front().first, ids.front());

    auto snapshot = index.export_snapshot();
    ASSERT_TRUE(snapshot.has_value());
    EXPECT_EQ(snapshot->kind, IndexSnapshotKind::gpu_ivf_flat);
    ASSERT_TRUE(snapshot->ivf.has_value());

    GpuIVFFlatIndex restored(backend, Metric::euclidean, kDimension, config);
    auto imported = restored.import_snapshot(*snapshot);
    ASSERT_TRUE(imported.has_value());

    const auto after = restored.search(query, 2);
    EXPECT_EQ(hit_ids(after), hit_ids(before));
}

TEST(GpuIVFFlatIndexTest, cpu_transfer_round_trip_rehydrates_exact_index) {
    test::RecordingBackend backend;
    const auto ids = sample_ids();
    const auto vectors = sample_vectors();
    auto config = make_ivf_config();

    ExactIndex cpu_source(Metric::euclidean, kDimension);
    populate_exact_index(cpu_source, ids, vectors);

    GpuIVFFlatIndex gpu_index(backend, Metric::euclidean, kDimension, config);
    auto imported = gpu_index.import_from_cpu_index(cpu_source);
    ASSERT_TRUE(imported.has_value());
    EXPECT_EQ(gpu_index.size(), ids.size());

    ExactIndex cpu_roundtrip(Metric::euclidean, kDimension);
    auto exported = gpu_index.export_to_cpu_index(cpu_roundtrip);
    ASSERT_TRUE(exported.has_value());
    EXPECT_EQ(cpu_roundtrip.size(), ids.size());

    const std::array<float, kDimension> query{5.2F, 5.1F, 5.0F, 5.0F};
    const auto hits = cpu_roundtrip.search(query, 1);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits.front().first, ids[3]);
}

TEST(GpuIVFPQIndexTest, snapshot_round_trip_preserves_top_hit) {
    test::RecordingBackend backend;
    const auto ids = sample_ids();
    const auto vectors = sample_vectors();
    auto config = make_ivf_config();

    GpuIVFPQIndex index(backend, Metric::euclidean, kDimension, config);
    auto built = index.build_from_batch(vectors, ids, GpuIndexBuildParams{});
    ASSERT_TRUE(built.has_value());

    const std::array<float, kDimension> query{-4.0F, -4.0F, -4.0F, -4.0F};
    const auto before = index.search(query, 2);
    ASSERT_FALSE(before.empty());
    EXPECT_EQ(before.front().first, ids[4]);

    auto snapshot = index.export_snapshot();
    ASSERT_TRUE(snapshot.has_value());
    EXPECT_EQ(snapshot->kind, IndexSnapshotKind::gpu_ivf_pq);
    ASSERT_TRUE(snapshot->ivf.has_value());
    ASSERT_TRUE(snapshot->pq.has_value());
    EXPECT_EQ(snapshot->pq->pq_dim, 2U);

    GpuIVFPQIndex restored(backend, Metric::euclidean, kDimension, config);
    auto imported = restored.import_snapshot(*snapshot);
    ASSERT_TRUE(imported.has_value());

    const auto after = restored.search(query, 2);
    ASSERT_FALSE(after.empty());
    EXPECT_EQ(after.front().first, before.front().first);
}

TEST(GpuGraphIndexTest, cpu_transfer_builds_real_graph_search_state) {
    test::RecordingBackend backend;
    const auto ids = sample_ids();
    const auto vectors = sample_vectors();
    auto config = make_ivf_config();
    config.graph_params.graph_degree = 4;
    config.graph_params.intermediate_graph_degree = 8;
    config.default_ef_search_gpu = 12;

    ExactIndex cpu_source(Metric::euclidean, kDimension);
    populate_exact_index(cpu_source, ids, vectors);

    GpuGraphIndex graph_index(backend, Metric::euclidean, kDimension, config);
    auto imported = graph_index.import_from_cpu_index(cpu_source);
    ASSERT_TRUE(imported.has_value());
    EXPECT_EQ(graph_index.size(), ids.size());

    const std::array<float, kDimension> query{0.0F, 0.0F, 0.0F, 0.0F};
    const auto hits = graph_index.search(query, 2);
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits.front().first, ids.front());

    ExactIndex roundtrip(Metric::euclidean, kDimension);
    auto exported = graph_index.export_to_cpu_index(roundtrip);
    ASSERT_TRUE(exported.has_value());
    EXPECT_EQ(roundtrip.size(), ids.size());
}

TEST(GpuHybridIndexTest, import_from_cpu_keeps_gpu_and_cpu_mirror_in_sync) {
    test::RecordingBackend backend;
    const auto ids = sample_ids();
    const auto vectors = sample_vectors();
    auto config = make_ivf_config();
    config.algorithm = GpuIndexAlgorithm::IvfFlat;

    ExactIndex cpu_source(Metric::euclidean, kDimension);
    populate_exact_index(cpu_source, ids, vectors);

    auto cpu_mirror = std::make_unique<ExactIndex>(Metric::euclidean, kDimension);
    GpuHybridIndex hybrid(
        backend, std::move(cpu_mirror), Metric::euclidean, kDimension, config);
    auto imported = hybrid.import_from_cpu_index(cpu_source);
    ASSERT_TRUE(imported.has_value());
    EXPECT_EQ(hybrid.size(), ids.size());

    const std::array<float, kDimension> query{5.0F, 5.0F, 5.0F, 5.0F};
    const auto hits = hybrid.search(query, 2);
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits.front().first, ids[2]);

    ExactIndex cpu_roundtrip(Metric::euclidean, kDimension);
    auto exported = hybrid.export_to_cpu_index(cpu_roundtrip);
    ASSERT_TRUE(exported.has_value());
    EXPECT_EQ(cpu_roundtrip.size(), ids.size());

    auto snapshot = hybrid.export_snapshot();
    ASSERT_TRUE(snapshot.has_value());
    EXPECT_EQ(snapshot->kind, IndexSnapshotKind::gpu_hybrid);
}

TEST(GpuDistributedIndexTest, sharded_index_merges_results_from_children) {
    test::RecordingBackend backend;
    const auto ids = sample_ids();
    const auto vectors = sample_vectors();
    GpuConfig config;

    std::vector<std::unique_ptr<GpuIndexPort>> children;
    children.push_back(make_brute_force_child(backend, config));
    children.push_back(make_brute_force_child(backend, config));

    GpuDistributedIndex distributed(std::move(children), DistributedMode::shard);
    auto built = distributed.build_from_batch(vectors, ids, GpuIndexBuildParams{});
    ASSERT_TRUE(built.has_value());
    EXPECT_EQ(distributed.size(), ids.size());

    const std::array<float, kDimension> query{-4.1F, -4.0F, -4.0F, -4.0F};
    const auto hits = distributed.search(query, 2);
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits.front().first, ids[5]);

    auto snapshot = distributed.export_snapshot();
    ASSERT_TRUE(snapshot.has_value());
    EXPECT_EQ(snapshot->kind, IndexSnapshotKind::gpu_distributed);

    std::vector<std::unique_ptr<GpuIndexPort>> restored_children;
    restored_children.push_back(make_brute_force_child(backend, config));
    restored_children.push_back(make_brute_force_child(backend, config));
    GpuDistributedIndex restored(std::move(restored_children), DistributedMode::shard);
    auto imported = restored.import_snapshot(*snapshot);
    ASSERT_TRUE(imported.has_value());

    const auto restored_hits = restored.search(query, 2);
    ASSERT_FALSE(restored_hits.empty());
    EXPECT_EQ(restored_hits.front().first, hits.front().first);
}

TEST(GpuDistributedIndexTest, replicated_index_keeps_logical_size_single_copy) {
    test::RecordingBackend backend;
    const auto ids = sample_ids();
    const auto vectors = sample_vectors();
    GpuConfig config;

    std::vector<std::unique_ptr<GpuIndexPort>> children;
    children.push_back(make_brute_force_child(backend, config));
    children.push_back(make_brute_force_child(backend, config));

    GpuDistributedIndex distributed(std::move(children), DistributedMode::replicate);
    auto built = distributed.build_from_batch(vectors, ids, GpuIndexBuildParams{});
    ASSERT_TRUE(built.has_value());

    EXPECT_EQ(distributed.size(), ids.size());
    const std::array<float, kDimension> query{0.1F, 0.1F, 0.0F, 0.0F};
    const auto hits = distributed.search(query, 1);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits.front().first, ids[1]);
}

}  // namespace
}  // namespace elips::gpu
