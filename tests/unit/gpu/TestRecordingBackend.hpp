#ifndef ELIPS_TESTS_UNIT_GPU_TEST_RECORDING_BACKEND_HPP
#define ELIPS_TESTS_UNIT_GPU_TEST_RECORDING_BACKEND_HPP

#include <algorithm>
#include <cstring>
#include <expected>
#include <utility>
#include <vector>

#include "elips/gpu_engine/GpuPort.hpp"
#include "elips/vector_engine/Metrics.hpp"

namespace elips::gpu::test {

class RecordingBackend final : public GpuPort {
public:
    RecordingBackend() {
        info_.name = "recording-backend";
        info_.vendor = "test";
        info_.backend = "test";
    }

    [[nodiscard]] std::expected<void, GpuError>
    initialize(const GpuConfig&) override {
        return {};
    }

    void shutdown() noexcept override {}

    [[nodiscard]] GpuDeviceInfo device_info() const noexcept override {
        return info_;
    }

    [[nodiscard]] bool is_available() const noexcept override { return true; }

    [[nodiscard]] std::expected<GpuBuffer, GpuError>
    allocate_device(size_t bytes) override {
        bytes_live_ += bytes;
        return GpuBuffer{nullptr, bytes, this};
    }

    void free_device(GpuBuffer buf) noexcept override {
        if (buf.backend_handle() != this) {
            return;
        }

        ++free_calls_;
        if (bytes_live_ >= buf.bytes()) {
            bytes_live_ -= buf.bytes();
        } else {
            bytes_live_ = 0;
        }
    }

    [[nodiscard]] std::expected<void, GpuError>
    upload(const void* host_src, GpuBuffer&, size_t bytes) override {
        const auto* begin = static_cast<const float*>(host_src);
        uploaded_.assign(begin, begin + bytes / sizeof(float));
        return {};
    }

    [[nodiscard]] std::expected<void, GpuError>
    download(const GpuBuffer&, void* host_dst, size_t bytes) override {
        if (uploaded_.empty()) {
            return {};
        }

        const size_t copy_bytes = std::min(bytes, uploaded_.size() * sizeof(float));
        std::memcpy(host_dst, uploaded_.data(), copy_bytes);
        return {};
    }

    [[nodiscard]] std::expected<void, GpuError>
    compute_distances_batch(std::span<const float> queries,
                            std::span<const float> database,
                            std::span<float> distances_out, size_t nq,
                            size_t nb, size_t dim,
                            elips::Metric metric) override {
        saw_expected_database_ = database.data() != nullptr &&
                                 database.size() == uploaded_.size() &&
                                 std::equal(database.begin(), database.end(),
                                            uploaded_.begin());

        for (size_t q = 0; q < nq; ++q) {
            const auto query = queries.subspan(q * dim, dim);
            for (size_t i = 0; i < nb; ++i) {
                const auto row = database.subspan(i * dim, dim);
                distances_out[q * nb + i] = elips::distance(metric, query, row);
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GpuError>
    top_k(std::span<const float> distances, std::span<uint32_t> indices_out,
          std::span<float> values_out, size_t nq, size_t nb,
          size_t k) override {
        for (size_t q = 0; q < nq; ++q) {
            std::vector<std::pair<float, uint32_t>> ranked;
            ranked.reserve(nb);
            for (size_t i = 0; i < nb; ++i) {
                ranked.emplace_back(distances[q * nb + i],
                                    static_cast<uint32_t>(i));
            }

            const auto take = std::min(k, ranked.size());
            std::partial_sort(
                ranked.begin(),
                ranked.begin() + static_cast<std::ptrdiff_t>(take),
                ranked.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

            for (size_t i = 0; i < take; ++i) {
                indices_out[q * k + i] = ranked[i].second;
                values_out[q * k + i] = ranked[i].first;
            }
        }
        return {};
    }

    void synchronize() override {}
    [[nodiscard]] bool is_idle() const noexcept override { return true; }

    void set_supports_ivf_pq(bool value) noexcept {
        info_.supports_ivf_pq = value;
    }

    [[nodiscard]] bool saw_expected_database() const noexcept {
        return saw_expected_database_;
    }

    [[nodiscard]] size_t free_calls() const noexcept { return free_calls_; }
    [[nodiscard]] size_t bytes_live() const noexcept { return bytes_live_; }

    void reset_observations() noexcept {
        saw_expected_database_ = false;
        uploaded_.clear();
    }

private:
    GpuDeviceInfo info_{};
    size_t free_calls_{0};
    size_t bytes_live_{0};
    bool saw_expected_database_{false};
    std::vector<float> uploaded_;
};

}  // namespace elips::gpu::test

#endif  // ELIPS_TESTS_UNIT_GPU_TEST_RECORDING_BACKEND_HPP
