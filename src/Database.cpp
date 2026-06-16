#include "elips/elips.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string_view>
#include <vector>

#include "elips/domain/Errors.hpp"
#include "elips/index_engine/IndexFactory.hpp"
#include "elips/kernel/LockManager.hpp"
#include "elips/storage/Serialization.hpp"
#include "elips/storage/WAL.hpp"
#include "elips/vector_engine/Metrics.hpp"

#ifdef ELIPS_GPU_ENABLED
#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/gpu_engine/GpuDeviceInfo.hpp"
#include "elips/gpu_engine/GpuMetricsSnapshot.hpp"
#include "elips/gpu_engine/GpuSelector.hpp"
#endif

namespace elips {
namespace {

namespace fs = std::filesystem;

using detail::get;
using detail::get_payload;
using detail::get_string;
using detail::put;
using detail::put_payload;
using detail::put_string;

constexpr std::uint32_t snapshot_magic = 0xE1105E01U;
constexpr std::uint32_t snapshot_version = 2U;
constexpr std::uint32_t identity_magic = 0xE11D0001U;
constexpr std::uint32_t text_embedder_magic = 0xE11D0002U;
constexpr const char* snapshot_file = "elips.snapshot";
constexpr const char* manifest_file = "elips.manifest";
constexpr const char* segment_dir = "segments";
constexpr std::uint32_t manifest_magic = 0xE1105E02U;
constexpr std::uint32_t segment_magic = 0xE1105E03U;
constexpr const char* identity_file = "IDENTITY";
constexpr const char* text_embedder_file = "TEXT_EMBEDDER.manifest";
constexpr const char* text_embedder_dir = "text_embedder";
constexpr const char* lock_file = "LOCK";
constexpr const char* wal_file = "wal.log";

struct Identity {
    std::uint16_t dimension{0};
    Metric metric{Metric::cosine};
    IndexType index{IndexType::graph};
};

struct SegmentManifestEntry {
    std::string vault_name;
    std::string file_name;
};

struct PersistedTextEmbedder {
    TextEmbedderInfo info;
    bool storage_path_relative{false};
};

bool all_finite(std::span<const float> values) noexcept {
    for (const float v : values) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> tokenize_terms(std::string_view text) {
    std::vector<std::string> tokens;
    std::string current;
    for (const unsigned char ch : text) {
        if (std::isalnum(ch) != 0) {
            current.push_back(static_cast<char>(std::tolower(ch)));
            continue;
        }
        if (!current.empty()) {
            tokens.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) {
        tokens.push_back(std::move(current));
    }
    return tokens;
}

float lexical_overlap_score(std::string_view query, std::string_view document) {
    const auto query_terms = tokenize_terms(query);
    if (query_terms.empty()) {
        return 0.0F;
    }
    const auto document_terms = tokenize_terms(document);
    if (document_terms.empty()) {
        return 0.0F;
    }
    std::set<std::string> query_set(query_terms.begin(), query_terms.end());
    std::set<std::string> document_set(document_terms.begin(),
                                       document_terms.end());
    std::vector<std::string> overlap;
    std::set_intersection(query_set.begin(), query_set.end(),
                          document_set.begin(), document_set.end(),
                          std::back_inserter(overlap));
    return static_cast<float>(overlap.size()) /
           static_cast<float>(query_set.size());
}

bool begins_with(std::string_view text, std::string_view prefix) noexcept {
    return text.size() >= prefix.size() &&
           text.substr(0, prefix.size()) == prefix;
}

SearchResult make_result(const Record& record, float distance_value) {
    return SearchResult{record.id, distance_value, record.payload, record.document,
                        record.chunk, record.lineage};
}

bool same_text_embedder_identity(const TextEmbedderInfo& lhs,
                                 const TextEmbedderInfo& rhs) {
    if (lhs.kind != rhs.kind || lhs.dimension != rhs.dimension ||
        lhs.provider != rhs.provider || lhs.model != rhs.model ||
        lhs.revision != rhs.revision) {
        return false;
    }
    if (!lhs.fingerprint.empty() && !rhs.fingerprint.empty()) {
        return lhs.fingerprint == rhs.fingerprint;
    }
    return true;
}

std::string storage_path_in_root(const fs::path& root, std::string_view model,
                                 std::string_view revision,
                                 const std::uint16_t dimension) {
    return (root / text_embedder_dir /
            (std::string(model) + "_" + std::string(revision) + "_" +
             std::to_string(dimension) + ".localembed"))
        .string();
}

PersistedTextEmbedder to_persisted_text_embedder(const fs::path& root,
                                                 TextEmbedderInfo info) {
    PersistedTextEmbedder persisted{.info = std::move(info)};
    if (!persisted.info.storage_path.empty()) {
        std::error_code absolute_ec;
        const fs::path absolute_storage =
            fs::absolute(persisted.info.storage_path, absolute_ec);
        const fs::path absolute_root = fs::absolute(root, absolute_ec);
        if (!absolute_ec) {
            std::error_code relative_ec;
            const fs::path relative_storage =
                fs::relative(absolute_storage, absolute_root, relative_ec);
            const auto candidate = relative_storage.generic_string();
            if (!relative_ec && !candidate.empty() &&
                !begins_with(candidate, "..")) {
                persisted.info.storage_path = candidate;
                persisted.storage_path_relative = true;
            }
        }
    }
    persisted.info.loaded = false;
    return persisted;
}

std::string resolve_persisted_storage_path(const fs::path& root,
                                          const PersistedTextEmbedder& persisted) {
    if (persisted.info.storage_path.empty()) {
        return {};
    }
    if (!persisted.storage_path_relative) {
        return persisted.info.storage_path;
    }
    return (root / persisted.info.storage_path).string();
}

void write_text_embedder_manifest(const fs::path& path,
                                  const PersistedTextEmbedder& persisted) {
    const fs::path tmp = path / (std::string(text_embedder_file) + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw StorageError{"cannot open text embedder manifest for writing"};
        }
        put<std::uint32_t>(out, text_embedder_magic);
        put<std::uint32_t>(out, snapshot_version);
        put<std::uint8_t>(out, static_cast<std::uint8_t>(persisted.info.kind));
        put<std::uint8_t>(out, persisted.storage_path_relative ? 1U : 0U);
        put<std::uint8_t>(out, persisted.info.rehydratable ? 1U : 0U);
        put<std::uint8_t>(out, persisted.info.auto_attached ? 1U : 0U);
        put<std::uint16_t>(out, persisted.info.dimension);
        put_string(out, persisted.info.provider);
        put_string(out, persisted.info.model);
        put_string(out, persisted.info.revision);
        put_string(out, persisted.info.backend);
        put_string(out, persisted.info.fingerprint);
        put_string(out, persisted.info.storage_path);
        if (!out) {
            throw StorageError{"error while writing text embedder manifest"};
        }
    }

    fs::rename(tmp, path / text_embedder_file);
}

std::optional<PersistedTextEmbedder> read_text_embedder_manifest(
    const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    if (get<std::uint32_t>(in) != text_embedder_magic) {
        throw StorageError{"text embedder manifest magic mismatch"};
    }
    const auto version = get<std::uint32_t>(in);
    if (version != snapshot_version) {
        throw StorageError{"unsupported text embedder manifest version"};
    }

    PersistedTextEmbedder persisted;
    persisted.info.kind =
        static_cast<TextEmbedderKind>(get<std::uint8_t>(in));
    persisted.storage_path_relative = get<std::uint8_t>(in) != 0U;
    persisted.info.rehydratable = get<std::uint8_t>(in) != 0U;
    persisted.info.auto_attached = get<std::uint8_t>(in) != 0U;
    persisted.info.dimension = get<std::uint16_t>(in);
    persisted.info.provider = get_string(in);
    persisted.info.model = get_string(in);
    persisted.info.revision = get_string(in);
    persisted.info.backend = get_string(in);
    persisted.info.fingerprint = get_string(in);
    persisted.info.storage_path = get_string(in);
    persisted.info.loaded = false;
    if (!in) {
        throw StorageError{"truncated text embedder manifest"};
    }
    return persisted;
}

std::string missing_text_embedder_message(const Config& config) {
    if (const auto info = config.text_embedder_info(); info.has_value()) {
        if (!info->rehydratable) {
            return "text embedder is not configured. This database expects '" +
                   info->provider + "/" + info->model +
                   "' but it cannot be rehydrated automatically. Reopen with "
                   "a matching embedder, configure a local embedder, or use "
                   "vector-first APIs such as place() and seek().";
        }
        return "text embedder is not configured. Configure a local text "
               "embedder or reopen with automatic text embedding enabled, or "
               "use vector-first APIs such as place() and seek().";
    }
    return "text embedder is not configured. Configure a local text embedder "
           "first, or use vector-first APIs such as place() and seek().";
}

std::optional<PersistedTextEmbedder> resolve_text_embedder_for_open(
    const fs::path& root, const bool persistent, const bool new_database,
    Config& effective) {
    const auto persisted =
        persistent ? read_text_embedder_manifest(root / text_embedder_file)
                   : std::optional<PersistedTextEmbedder>{};

    if (effective.text_embedder() != nullptr) {
        effective.text_embedder()->set_output_dimension(effective.dimension());
        TextEmbedderInfo info = effective.text_embedder()->info(false);
        if (info.dimension == 0U) {
            info.dimension = effective.dimension();
        }
        if (info.dimension != effective.dimension()) {
            throw ConfigError{
                "configured text embedder dimension conflicts with database"};
        }

        if (persisted.has_value()) {
            TextEmbedderInfo persisted_info = persisted->info;
            persisted_info.storage_path =
                resolve_persisted_storage_path(root, *persisted);
            if (!same_text_embedder_identity(info, persisted_info)) {
                throw ConfigError{
                    "configured text embedder conflicts with database"};
            }
            info.auto_attached = persisted_info.auto_attached;
            if (info.storage_path.empty()) {
                info.storage_path = persisted_info.storage_path;
            }
        }

        effective.attach_runtime_text_embedder(effective.text_embedder(), info);
        if (!persistent || persisted.has_value() ||
            effective.access_mode() == AccessMode::read_only) {
            return std::nullopt;
        }
        return to_persisted_text_embedder(root, info);
    }

    if (effective.has_pending_local_text_embedder()) {
        auto options = *effective.local_text_embedder_options();
        if (options.dimension == 0U) {
            options.dimension = effective.dimension();
        }
        if (options.storage_path.empty() && persistent) {
            options.storage_path =
                storage_path_in_root(root, options.model, options.revision,
                                     options.dimension);
        }

        TextEmbedderInfo info =
            describe_local_text_embedder(options, effective.dimension(), false);
        if (info.dimension != effective.dimension()) {
            throw ConfigError{
                "configured local text embedder dimension conflicts with database"};
        }

        if (persisted.has_value()) {
            TextEmbedderInfo persisted_info = persisted->info;
            persisted_info.storage_path =
                resolve_persisted_storage_path(root, *persisted);
            if (!same_text_embedder_identity(info, persisted_info) ||
                persisted_info.kind != TextEmbedderKind::local_builtin) {
                throw ConfigError{
                    "configured local text embedder conflicts with database"};
            }
            if (!options.storage_path.empty() &&
                options.storage_path != persisted_info.storage_path) {
                throw ConfigError{
                    "configured local text embedder storage path conflicts "
                    "with database"};
            }
            options.storage_path = persisted_info.storage_path;
            if (!fs::exists(options.storage_path)) {
                throw StorageError{"local text embedder artifact is missing: " +
                                   options.storage_path};
            }
        }

        auto runtime = make_local_text_embedder(options);
        info = runtime->info(false);
        effective.attach_runtime_text_embedder(std::move(runtime), info);
        if (!persistent || persisted.has_value() ||
            effective.access_mode() == AccessMode::read_only) {
            return std::nullopt;
        }
        return to_persisted_text_embedder(root, info);
    }

    if (persisted.has_value()) {
        TextEmbedderInfo persisted_info = persisted->info;
        persisted_info.storage_path =
            resolve_persisted_storage_path(root, *persisted);

        if (effective.auto_text_embedder() &&
            persisted_info.kind == TextEmbedderKind::local_builtin &&
            persisted_info.rehydratable) {
            if (persisted_info.dimension != effective.dimension()) {
                throw ConfigError{
                    "database text embedder dimension conflicts with database"};
            }
            if (!fs::exists(persisted_info.storage_path)) {
                throw StorageError{"local text embedder artifact is missing: " +
                                   persisted_info.storage_path};
            }
            auto runtime = make_local_text_embedder(LocalTextEmbedderOptions{
                .model = persisted_info.model,
                .revision = persisted_info.revision,
                .storage_path = persisted_info.storage_path,
                .dimension = persisted_info.dimension,
            });
            TextEmbedderInfo info = runtime->info(persisted_info.auto_attached);
            effective.attach_runtime_text_embedder(std::move(runtime), info);
            return std::nullopt;
        }

        effective.remember_expected_text_embedder(std::move(persisted_info));
        return std::nullopt;
    }

    if ((persistent && !new_database) || !effective.auto_text_embedder()) {
        return std::nullopt;
    }

    LocalTextEmbedderOptions options{
        .model = "default",
        .revision = "v1",
        .storage_path = persistent
                            ? storage_path_in_root(root, "default", "v1",
                                                   effective.dimension())
                            : std::string{},
        .dimension = effective.dimension(),
    };
    auto runtime = make_local_text_embedder(options);
    TextEmbedderInfo info = runtime->info(true);
    effective.attach_runtime_text_embedder(std::move(runtime), info);
    if (!persistent || effective.access_mode() == AccessMode::read_only) {
        return std::nullopt;
    }
    return to_persisted_text_embedder(root, info);
}

void put_record(std::ostream& out, const Record& record, bool with_extras = true) {
    out.write(reinterpret_cast<const char*>(record.id.bytes().data()),
              static_cast<std::streamsize>(record.id.bytes().size()));
    const auto values = record.vector.values();
    put<std::uint16_t>(out, static_cast<std::uint16_t>(values.size()));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size_bytes()));
    put_payload(out, record.payload);
    if (!with_extras) {
        return;
    }
    detail::put_document_attachment(out, record.document);
    detail::put_chunk_info(out, record.chunk);
    detail::put_embedding_lineage(out, record.lineage);
}

Record get_record(std::istream& in, bool with_extras) {
    RecordID::Bytes bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    const auto dim = get<std::uint16_t>(in);
    std::vector<float> values(dim);
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(dim) * sizeof(float));
    Payload payload = get_payload(in);

    std::optional<DocumentAttachment> document;
    std::optional<ChunkInfo> chunk;
    std::optional<EmbeddingLineage> lineage;
    if (with_extras) {
        document = detail::get_document_attachment(in);
        chunk = detail::get_chunk_info(in);
        lineage = detail::get_embedding_lineage(in);
    }

    if (!in) {
        throw StorageError{"truncated or corrupt record payload"};
    }

    return Record{RecordID{bytes}, Vector{std::move(values)}, std::move(payload),
                  std::move(document), std::move(chunk), std::move(lineage)};
}

void write_identity(const fs::path& file, const Config& config) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw StorageError{"cannot write IDENTITY"};
    }
    put<std::uint32_t>(out, identity_magic);
    put<std::uint32_t>(out, snapshot_version);
    put<std::uint16_t>(out, config.dimension());
    put<std::uint8_t>(out, static_cast<std::uint8_t>(config.metric()));
    put<std::uint8_t>(out, static_cast<std::uint8_t>(config.index()));
}

Identity read_identity(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in || get<std::uint32_t>(in) != identity_magic) {
        throw StorageError{"corrupt IDENTITY"};
    }
    (void)get<std::uint32_t>(in);  // version
    Identity id;
    id.dimension = get<std::uint16_t>(in);
    id.metric = static_cast<Metric>(get<std::uint8_t>(in));
    id.index = static_cast<IndexType>(get<std::uint8_t>(in));
    return id;
}

void write_snapshot_file(
    const fs::path& path, const Config& config,
    const std::map<std::string, std::unique_ptr<Vault>>& vaults) {
    const fs::path tmp = path / (std::string(snapshot_file) + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw StorageError{"cannot open snapshot for writing"};
        }
        put<std::uint32_t>(out, snapshot_magic);
        put<std::uint32_t>(out, snapshot_version);
        put<std::uint16_t>(out, config.dimension());
        put<std::uint8_t>(out, static_cast<std::uint8_t>(config.metric()));
        put<std::uint32_t>(out, static_cast<std::uint32_t>(vaults.size()));
        for (const auto& [name, vault] : vaults) {
            put_string(out, name);
            const auto& records = vault->records();
            put<std::uint32_t>(out, static_cast<std::uint32_t>(records.size()));
            for (const auto& [id, record] : records) {
                (void)id;
                put_record(out, record);
            }
        }
        if (!out) {
            throw StorageError{"error while writing snapshot"};
        }
    }

    fs::rename(tmp, path / snapshot_file);
}

void write_segment_file(const fs::path& file, const std::string& vault_name,
                        const Vault& vault) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw StorageError{"cannot write vault segment"};
    }
    put<std::uint32_t>(out, segment_magic);
    put<std::uint32_t>(out, snapshot_version);
    put_string(out, vault_name);
    const auto& records = vault.records();
    put<std::uint32_t>(out, static_cast<std::uint32_t>(records.size()));
    for (const auto& [id, record] : records) {
        (void)id;
        put_record(out, record);
    }
    if (!out) {
        throw StorageError{"error while writing vault segment"};
    }
}

void write_manifest_file(const fs::path& path, const Config& config,
                         const std::vector<SegmentManifestEntry>& entries) {
    const fs::path tmp = path / (std::string(manifest_file) + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw StorageError{"cannot open manifest for writing"};
        }
        put<std::uint32_t>(out, manifest_magic);
        put<std::uint32_t>(out, snapshot_version);
        put<std::uint16_t>(out, config.dimension());
        put<std::uint8_t>(out, static_cast<std::uint8_t>(config.metric()));
        put<std::uint32_t>(out, static_cast<std::uint32_t>(entries.size()));
        for (const auto& entry : entries) {
            put_string(out, entry.vault_name);
            put_string(out, entry.file_name);
        }
        if (!out) {
            throw StorageError{"error while writing manifest"};
        }
    }

    fs::rename(tmp, path / manifest_file);
}

std::vector<SegmentManifestEntry> read_manifest_file(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw StorageError{"cannot open manifest for reading"};
    }
    if (get<std::uint32_t>(in) != manifest_magic) {
        throw StorageError{"manifest magic mismatch"};
    }
    const auto version = get<std::uint32_t>(in);
    if (version != snapshot_version) {
        throw StorageError{"unsupported manifest version"};
    }
    (void)get<std::uint16_t>(in);
    (void)get<std::uint8_t>(in);

    const auto count = get<std::uint32_t>(in);
    std::vector<SegmentManifestEntry> entries;
    entries.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        entries.push_back(
            SegmentManifestEntry{get_string(in), get_string(in)});
    }
    return entries;
}

void load_snapshot_file(const fs::path& file, ElipsInstance& instance) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw StorageError{"cannot open snapshot for reading"};
    }
    if (get<std::uint32_t>(in) != snapshot_magic) {
        throw StorageError{"snapshot magic mismatch"};
    }
    const auto version = get<std::uint32_t>(in);
    if (version != 1U && version != snapshot_version) {
        throw StorageError{"unsupported snapshot version"};
    }
    (void)get<std::uint16_t>(in);
    (void)get<std::uint8_t>(in);

    const auto vault_count = get<std::uint32_t>(in);
    for (std::uint32_t vault_index = 0; vault_index < vault_count;
         ++vault_index) {
        const std::string name = get_string(in);
        Vault& vault = instance.vault(name);
        vault.set_read_only(false);
        const auto record_count = get<std::uint32_t>(in);
        for (std::uint32_t record_index = 0; record_index < record_count;
             ++record_index) {
            const Record record = get_record(in, version >= snapshot_version);
            vault.place(record.vector, record.payload, record.id, record.document,
                        record.chunk, record.lineage);
        }
    }
}

void load_segmented_state(const fs::path& root, ElipsInstance& instance) {
    const auto entries = read_manifest_file(root / manifest_file);
    for (const auto& entry : entries) {
        const fs::path segment_path = root / segment_dir / entry.file_name;
        std::ifstream in(segment_path, std::ios::binary);
        if (!in) {
            throw StorageError{"missing vault segment: " + entry.file_name};
        }
        if (get<std::uint32_t>(in) != segment_magic) {
            throw StorageError{"segment magic mismatch"};
        }
        const auto version = get<std::uint32_t>(in);
        if (version != snapshot_version) {
            throw StorageError{"unsupported segment version"};
        }
        const std::string stored_name = get_string(in);
        if (stored_name != entry.vault_name) {
            throw StorageError{"segment vault mismatch for " + entry.file_name};
        }
        const auto record_count = get<std::uint32_t>(in);
        Vault& vault = instance.vault(entry.vault_name);
        vault.set_read_only(false);
        for (std::uint32_t record_index = 0; record_index < record_count;
             ++record_index) {
            const Record record = get_record(in, true);
            vault.place(record.vector, record.payload, record.id, record.document,
                        record.chunk, record.lineage);
        }
    }
}

void configure_gpu_backend(ElipsInstance& instance, const Config& config) {
#ifdef ELIPS_GPU_ENABLED
    gpu::GpuDeviceManager manager;
    instance.set_gpu_info(manager.cpu_fallback_info());

    if (!config.has_gpu() || config.gpu().policy == gpu::GpuPolicy::CpuOnly) {
        return;
    }

    const auto devices = manager.probe_all_devices();
    if (devices.empty()) {
        if (config.gpu().policy == gpu::GpuPolicy::RequireGpu ||
            config.gpu().policy == gpu::GpuPolicy::Specific) {
            throw ConfigError{"GPU acceleration was requested, but no compatible "
                              "device was found"};
        }
        return;
    }

    gpu::GpuSelector selector;
    auto selected = selector.select(config.gpu(), devices);
    if (!selected.has_value() || *selected == nullptr) {
        if (config.gpu().policy == gpu::GpuPolicy::RequireGpu ||
            config.gpu().policy == gpu::GpuPolicy::Specific) {
            throw ConfigError{"GPU acceleration was requested, but the backend "
                              "could not be initialized"};
        }
        return;
    }

    instance.set_gpu_available(true);
    instance.set_gpu_info((*selected)->device_info());
    instance.set_gpu_backend(std::move(*selected));
#else
    (void)instance;
    (void)config;
#endif
}

void apply_read_only_mode(ElipsInstance& instance) {
    for (const auto& name : instance.list_vaults()) {
        instance.vault(name).set_read_only(true);
    }
}

}  // namespace

// --------------------------------- Vault ---------------------------------

Vault::Vault(std::string name, const Config& config
#ifdef ELIPS_GPU_ENABLED
             ,
             gpu::GpuPort* gpu_backend
#endif
             )
    : name_(std::move(name)),
      config_(config),
      index_(make_index(config, config.dimension()
#ifdef ELIPS_GPU_ENABLED
                        ,
                        gpu_backend
#endif
                        ))
#ifdef ELIPS_GPU_ENABLED
      ,
      gpu_backend_(gpu_backend)
#endif
{
}

Vector Vault::prepare(const Vector& vector) const {
    if (vector.dimension() != config_.dimension()) {
        throw DimensionMismatch{"vector dimension does not match vault"};
    }
    if (!all_finite(vector.values())) {
        throw InvalidVector{"vector contains NaN or Inf"};
    }
    return requires_normalization(config_.metric()) ? vector.normalized()
                                                    : vector;
}

void Vault::ensure_writable() const {
    if (read_only_) {
        throw StorageError{"vault is opened in read-only mode"};
    }
}

RecordID Vault::place(const Vector& vector, Payload payload,
                      std::optional<RecordID> id,
                      std::optional<DocumentAttachment> document,
                      std::optional<ChunkInfo> chunk,
                      std::optional<EmbeddingLineage> lineage) {
    ensure_writable();

    Vector prepared = prepare(vector);
    const RecordID record_id = id.value_or(RecordID::generate());

    if (wal_ != nullptr) {
        wal_->append_insert(name_, record_id, prepared.values(), payload, document,
                            chunk, lineage);
    }

    const auto existing = records_.find(record_id);
    if (existing != records_.end()) {
        if (config_.metadata_acceleration()) {
            metadata_index_.remove(record_id, existing->second.payload);
        }
        index_->remove(record_id);
    }

    index_->insert(record_id, prepared.values());
    if (config_.metadata_acceleration()) {
        metadata_index_.insert(record_id, payload);
    }

    records_[record_id] =
        Record{record_id, std::move(prepared), std::move(payload),
               std::move(document), std::move(chunk), std::move(lineage)};
    return record_id;
}

RecordID Vault::place_document(std::string text, Payload payload,
                               std::optional<RecordID> id,
                               std::optional<ChunkInfo> chunk,
                               std::optional<EmbeddingLineage> lineage) {
    if (!config_.has_text_embedder()) {
        throw ConfigError{missing_text_embedder_message(config_)};
    }

    const RecordID record_id = id.value_or(RecordID::generate());
    DocumentAttachment document{
        .text = text,
        .uri = {},
        .mime_type = "text/plain",
    };

    if (!chunk.has_value()) {
        chunk = ChunkInfo{.document_key = record_id.to_string()};
    } else if (chunk->document_key.empty()) {
        chunk->document_key = record_id.to_string();
    }

    if (!lineage.has_value()) {
        auto info = config_.text_embedder()->info();
        Payload attributes;
        if (!info.backend.empty()) {
            attributes.emplace("backend", info.backend);
        }
        if (!info.fingerprint.empty()) {
            attributes.emplace("fingerprint", info.fingerprint);
        }
        lineage = EmbeddingLineage{
            .provider = std::move(info.provider),
            .model = std::move(info.model),
            .revision = std::move(info.revision),
            .attributes = std::move(attributes),
        };
    }

    return place(config_.text_embedder()->embed(text), std::move(payload), record_id,
                 std::move(document), std::move(chunk), std::move(lineage));
}

void Vault::place_many(const std::vector<Record>& records) {
    for (const auto& record : records) {
        const std::optional<RecordID> id =
            (record.id == RecordID{}) ? std::nullopt
                                      : std::optional<RecordID>{record.id};
        place(record.vector, record.payload, id, record.document, record.chunk,
              record.lineage);
    }
}

std::vector<SearchResult> Vault::search_records(
    const Vector& prepared, std::size_t top, const Filter& filter,
    std::optional<float> threshold,
    const std::vector<const Record*>* subset) const {
    if (top == 0) {
        return {};
    }

    std::vector<SearchResult> results;
    const auto reserve =
        subset != nullptr ? subset->size() : static_cast<std::size_t>(records_.size());
    results.reserve(std::min(reserve, top > 0 ? top * 4 : reserve));

    const auto accumulate = [&](const Record& record) {
        if (!filter.matches(record.payload)) {
            return;
        }
        const float distance_value =
            distance(config_.metric(), prepared.values(), record.vector.values());
        if (threshold.has_value() && distance_value > *threshold) {
            return;
        }
        results.push_back(make_result(record, distance_value));
    };

    if (subset != nullptr) {
        for (const Record* record : *subset) {
            if (record != nullptr) {
                accumulate(*record);
            }
        }
    } else {
        for (const auto& [id, record] : records_) {
            (void)id;
            accumulate(record);
        }
    }

    const auto ordering = [](const SearchResult& lhs, const SearchResult& rhs) {
        return lhs.distance < rhs.distance;
    };
    if (results.size() > top) {
        std::partial_sort(results.begin(),
                          results.begin() + static_cast<std::ptrdiff_t>(top),
                          results.end(), ordering);
        results.resize(top);
    } else {
        std::sort(results.begin(), results.end(), ordering);
    }
    return results;
}

QueryPlan Vault::plan_seek(const Vector& prepared, std::size_t top,
                           const Filter& filter,
                           std::optional<float> threshold,
                           bool has_text_component) const {
    (void)prepared;

    QueryPlan plan;
    plan.index_type = std::string(index_->type_name());
    plan.gpu_index = begins_with(plan.index_type, "gpu_");
    plan.candidate_count = records_.size();

    if (records_.empty() || top == 0) {
        plan.strategy =
            has_text_component ? QueryStrategy::text_probe : QueryStrategy::full_scan;
        return plan;
    }

    const bool prefer_full_scan =
        threshold.has_value() || records_.size() <= 128 ||
        top >= std::max<std::size_t>(records_.size() / 2, 1U);

    if (config_.metadata_acceleration() && !filter.matches_all()) {
        if (auto candidates = metadata_index_.exact_candidates(filter);
            candidates.has_value()) {
            plan.metadata_accelerated = true;
            plan.candidate_count = candidates->size();
            if (candidates->empty() ||
                candidates->size() <= std::max<std::size_t>(top * 8, 128U) ||
                prefer_full_scan) {
                plan.strategy = has_text_component ? QueryStrategy::hybrid_fusion
                                                   : QueryStrategy::exact_candidates;
                return plan;
            }
        }
    }

    if (prefer_full_scan) {
        plan.strategy =
            has_text_component ? QueryStrategy::hybrid_fusion : QueryStrategy::full_scan;
        return plan;
    }

    plan.strategy =
        has_text_component ? QueryStrategy::hybrid_fusion : QueryStrategy::ann_index;
    return plan;
}

std::vector<SearchResult> Vault::seek(const Vector& query, std::size_t top,
                                      const Filter& filter,
                                      std::optional<float> threshold) const {
    if (top == 0 || records_.empty()) {
        return {};
    }

    const Vector prepared = prepare(query);
    const QueryPlan plan = plan_seek(prepared, top, filter, threshold, false);

    if (plan.strategy == QueryStrategy::exact_candidates) {
        const auto candidates = metadata_index_.exact_candidates(filter);
        if (!candidates.has_value() || candidates->empty()) {
            return {};
        }
        std::vector<const Record*> subset;
        subset.reserve(candidates->size());
        for (const auto& id : *candidates) {
            const auto it = records_.find(id);
            if (it != records_.end()) {
                subset.push_back(&it->second);
            }
        }
        return search_records(prepared, top, filter, threshold, &subset);
    }

    if (plan.strategy == QueryStrategy::full_scan) {
        return search_records(prepared, top, filter, threshold);
    }

    std::size_t fetch = top;
    if (threshold.has_value()) {
        fetch = records_.size();
    } else if (!filter.matches_all()) {
        fetch = std::min(records_.size(), std::max<std::size_t>(top * 20, 64U));
    }
    fetch = std::max<std::size_t>(fetch, 1U);

    const auto hits = index_->search(prepared.values(), fetch);
    std::vector<SearchResult> results;
    results.reserve(std::min(hits.size(), top));
    for (const auto& [id, dist] : hits) {
        if (threshold.has_value() && dist > *threshold) {
            continue;
        }
        const auto it = records_.find(id);
        if (it == records_.end()) {
            continue;
        }
        if (!filter.matches(it->second.payload)) {
            continue;
        }
        results.push_back(make_result(it->second, dist));
        if (results.size() >= top) {
            break;
        }
    }
    return results;
}

std::vector<SearchResult> Vault::seek_text(std::string_view text, std::size_t top,
                                           const Filter& filter,
                                           std::optional<float> threshold) const {
    if (!config_.has_text_embedder()) {
        throw ConfigError{missing_text_embedder_message(config_)};
    }
    if (top == 0 || records_.empty()) {
        return {};
    }
    return seek(config_.text_embedder()->embed(text), top, filter, threshold);
}

std::vector<SearchResult> Vault::seek_hybrid(const Vector& query,
                                             std::string_view text,
                                             std::size_t top,
                                             const Filter& filter,
                                             std::optional<float> threshold,
                                             float lexical_weight) const {
    if (top == 0 || records_.empty()) {
        return {};
    }

    const float weight = std::clamp(lexical_weight, 0.0F, 1.0F);
    if (text.empty() || weight == 0.0F) {
        return seek(query, top, filter, threshold);
    }

    const std::size_t candidate_top =
        std::min(records_.size(), std::max<std::size_t>(top * 5, top));
    auto candidates = seek(query, candidate_top, filter, threshold);
    for (auto& result : candidates) {
        const auto it = records_.find(result.id);
        const float lexical_score =
            (it != records_.end() && it->second.document.has_value())
                ? lexical_overlap_score(text, it->second.document->text)
                : 0.0F;
        result.distance =
            ((1.0F - weight) * result.distance) + (weight * (1.0F - lexical_score));
    }

    const auto ordering = [](const SearchResult& lhs, const SearchResult& rhs) {
        return lhs.distance < rhs.distance;
    };
    if (candidates.size() > top) {
        std::partial_sort(candidates.begin(),
                          candidates.begin() +
                              static_cast<std::ptrdiff_t>(top),
                          candidates.end(), ordering);
        candidates.resize(top);
    } else {
        std::sort(candidates.begin(), candidates.end(), ordering);
    }
    return candidates;
}

QueryPlan Vault::explain_seek(const Vector& query, std::size_t top,
                              const Filter& filter,
                              std::optional<float> threshold,
                              bool has_text_component) const {
    return plan_seek(prepare(query), top, filter, threshold, has_text_component);
}

std::vector<Record> Vault::scan(const Filter& filter, std::size_t offset,
                                std::size_t limit) const {
    std::vector<Record> out;
    std::size_t skipped = 0;
    for (const auto& [id, record] : records_) {
        (void)id;
        if (!filter.matches(record.payload)) {
            continue;
        }
        if (skipped < offset) {
            ++skipped;
            continue;
        }
        if (out.size() >= limit) {
            break;
        }
        out.push_back(record);
    }
    return out;
}

std::optional<Record> Vault::fetch(const RecordID& id) const {
    const auto it = records_.find(id);
    if (it == records_.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool Vault::erase(const RecordID& id) {
    ensure_writable();

    const auto it = records_.find(id);
    if (it == records_.end()) {
        return false;
    }
    if (wal_ != nullptr) {
        wal_->append_erase(name_, id);
    }
    if (config_.metadata_acceleration()) {
        metadata_index_.remove(id, it->second.payload);
    }
    index_->remove(id);
    records_.erase(it);
    return true;
}

void Vault::rebuild_index() {
    ensure_writable();

    auto rebuilt = make_index(config_, config_.dimension()
#ifdef ELIPS_GPU_ENABLED
                              ,
                              gpu_backend_
#endif
    );

    metadata_index_ = MetadataIndex{};
    for (const auto& [id, record] : records_) {
        rebuilt->insert(id, record.vector.values());
        if (config_.metadata_acceleration()) {
            metadata_index_.insert(id, record.payload);
        }
    }
    index_ = std::move(rebuilt);
}

VaultInfo Vault::info() const noexcept {
    return VaultInfo{records_.size(), config_.dimension(), config_.metric()};
}

// ----------------------------- ElipsInstance -----------------------------

ElipsInstance::ElipsInstance(std::string path, Config config, bool persistent,
                             std::optional<LockManager> lock)
    : path_(std::move(path)),
      config_(config),
      persistent_(persistent),
      lock_(std::move(lock)) {}

ElipsInstance::~ElipsInstance() {
    if (persistent_ && !closed_ && config_.access_mode() != AccessMode::read_only) {
        try {
            checkpoint();
        } catch (...) {
            // E.16: destructors must not throw. Best-effort checkpoint.
        }
    }

    // Vault-owned GPU indexes hold non-owning backend references, so release
    // every vault before member destruction reaches gpu_backend_.
    vaults_.clear();
}

Vault& ElipsInstance::vault(const std::string& name) {
    const auto it = vaults_.find(name);
    if (it != vaults_.end()) {
        return *it->second;
    }

#ifdef ELIPS_GPU_ENABLED
    auto created = std::make_unique<Vault>(name, config_, gpu_backend_.get());
#else
    auto created = std::make_unique<Vault>(name, config_);
#endif
    created->set_wal(wal_.get());
    if (config_.access_mode() == AccessMode::read_only) {
        created->set_read_only(true);
    }

    Vault& ref = *created;
    vaults_.emplace(name, std::move(created));
    return ref;
}

Vault& ElipsInstance::adopt_vault(std::unique_ptr<Vault> vault) {
    vault->set_wal(wal_.get());
    if (config_.access_mode() == AccessMode::read_only) {
        vault->set_read_only(true);
    }
    Vault& ref = *vault;
    vaults_[vault->name()] = std::move(vault);
    return ref;
}

void ElipsInstance::attach_wal(std::unique_ptr<WAL> wal) {
    wal_ = std::move(wal);
    for (auto& [name, vault] : vaults_) {
        (void)name;
        vault->set_wal(wal_.get());
    }
}

std::vector<std::string> ElipsInstance::list_vaults() const {
    std::vector<std::string> names;
    names.reserve(vaults_.size());
    for (const auto& [name, vault] : vaults_) {
        (void)vault;
        names.push_back(name);
    }
    return names;
}

void ElipsInstance::checkpoint() {
    if (!persistent_ || config_.access_mode() == AccessMode::read_only) {
        return;
    }

    fs::create_directories(path_);
    const fs::path root = path_;

    if (config_.segmented_storage()) {
        const fs::path segments_root = root / segment_dir;
        fs::create_directories(segments_root);

        const auto epoch = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        std::vector<SegmentManifestEntry> entries;
        entries.reserve(vaults_.size());

        std::size_t ordinal = 0;
        for (const auto& [name, vault] : vaults_) {
            const std::string file_name =
                "vault_" + std::to_string(ordinal++) + "_" +
                std::to_string(epoch) + ".segment";
            const fs::path tmp_path = segments_root / (file_name + ".tmp");
            const fs::path final_path = segments_root / file_name;
            write_segment_file(tmp_path, name, *vault);
            fs::rename(tmp_path, final_path);
            entries.push_back(SegmentManifestEntry{name, file_name});
        }

        write_manifest_file(root, config_, entries);

        const std::set<std::string> keep_files = [&entries] {
            std::set<std::string> keep;
            for (const auto& entry : entries) {
                keep.insert(entry.file_name);
            }
            return keep;
        }();

        std::error_code ec;
        if (fs::exists(segments_root, ec)) {
            for (const auto& dir_entry : fs::directory_iterator(segments_root, ec)) {
                if (ec || !dir_entry.is_regular_file()) {
                    continue;
                }
                const auto file_name = dir_entry.path().filename().string();
                if (!keep_files.contains(file_name)) {
                    fs::remove(dir_entry.path(), ec);
                }
            }
        }
        fs::remove(root / snapshot_file, ec);
    } else {
        write_snapshot_file(root, config_, vaults_);
        std::error_code ec;
        fs::remove(root / manifest_file, ec);
        fs::remove_all(root / segment_dir, ec);
    }

    if (wal_ != nullptr) {
        wal_->reset();
    }
}

void ElipsInstance::compact() {
    if (!persistent_ || config_.access_mode() == AccessMode::read_only) {
        return;
    }
    for (auto& [name, vault] : vaults_) {
        (void)name;
        vault->rebuild_index();
    }
    checkpoint();
}

void ElipsInstance::close() {
    if (closed_) {
        return;
    }

    checkpoint();
    for (auto& [name, vault] : vaults_) {
        (void)name;
        vault->set_wal(nullptr);
    }
    wal_.reset();
    lock_.reset();
    closed_ = true;
}

// --------------------------------- open ----------------------------------

std::unique_ptr<ElipsInstance> open(const std::string& path,
                                    const Config& config) {
    if (path == ":memory:") {
        if (config.dimension() == 0) {
            throw ConfigError{"in-memory database requires a dimension"};
        }

        Config effective = config;
        (void)resolve_text_embedder_for_open(fs::path{}, /*persistent=*/false,
                                             /*new_database=*/true, effective);
        auto instance =
            std::make_unique<ElipsInstance>(path, effective, /*persistent=*/false);
        configure_gpu_backend(*instance, effective);
        if (effective.access_mode() == AccessMode::read_only) {
            apply_read_only_mode(*instance);
        }
        return instance;
    }

    const bool existing_database =
        fs::exists(fs::path(path) / identity_file);
    fs::create_directories(path);
    const LockMode lock_mode =
        config.access_mode() == AccessMode::read_only ? LockMode::shared
                                                      : LockMode::exclusive;
    LockManager lock{(fs::path(path) / lock_file).string(), lock_mode};

    const fs::path identity = fs::path(path) / identity_file;
    Config effective = config;
    if (existing_database) {
        const Identity id = read_identity(identity);
        if (config.dimension() != 0 && config.dimension() != id.dimension) {
            throw ConfigError{"configured dimension conflicts with database"};
        }
        effective.dimension(id.dimension).metric(id.metric).index(id.index);
    } else {
        if (config.access_mode() == AccessMode::read_only) {
            throw ConfigError{"read-only open requires an existing database"};
        }
        if (config.dimension() == 0) {
            throw ConfigError{"new database requires a dimension"};
        }
        write_identity(identity, config);
    }

    const auto embedder_manifest =
        resolve_text_embedder_for_open(fs::path(path), /*persistent=*/true,
                                       !existing_database, effective);
    if (embedder_manifest.has_value() &&
        effective.access_mode() != AccessMode::read_only) {
        write_text_embedder_manifest(fs::path(path), *embedder_manifest);
    }

    auto instance = std::make_unique<ElipsInstance>(path, effective,
                                                    /*persistent=*/true,
                                                    std::move(lock));
    configure_gpu_backend(*instance, effective);

    const fs::path manifest = fs::path(path) / manifest_file;
    const fs::path snapshot = fs::path(path) / snapshot_file;
    if (fs::exists(manifest)) {
        load_segmented_state(path, *instance);
    } else if (fs::exists(snapshot)) {
        load_snapshot_file(snapshot, *instance);
    }

    const fs::path walpath = fs::path(path) / wal_file;
    for (const auto& entry : WAL::replay(walpath)) {
        Vault& vault = instance->vault(entry.vault);
        vault.set_read_only(false);
        if (entry.op == WAL::Op::insert || entry.op == WAL::Op::insert_ex) {
            vault.place(Vector{entry.vector}, entry.payload, entry.id, entry.document,
                        entry.chunk, entry.lineage);
        } else {
            vault.erase(entry.id);
        }
    }

    if (effective.access_mode() != AccessMode::read_only &&
        effective.durability() != Durability::ephemeral) {
        const bool sync = effective.durability() != Durability::relaxed;
        instance->attach_wal(std::make_unique<WAL>(walpath, sync));
    }

    if (effective.access_mode() == AccessMode::read_only) {
        apply_read_only_mode(*instance);
    }

    return instance;
}

// ------------------------------ Transaction ------------------------------

Transaction ElipsInstance::begin_transaction() { return Transaction{*this}; }

Transaction::~Transaction() {
    if (!done_) {
        rollback();
    }
}

void Transaction::enqueue_place(std::string vault, const Vector& vector,
                                Payload payload,
                                std::optional<RecordID> id) {
    if (vector.dimension() != db_->config().dimension()) {
        throw DimensionMismatch{"vector dimension does not match database"};
    }
    if (!all_finite(vector.values())) {
        throw InvalidVector{"vector contains NaN or Inf"};
    }
    ops_.push_back(PendingOp{false, std::move(vault), vector, std::move(payload),
                             std::move(id)});
}

void Transaction::enqueue_erase(std::string vault, const RecordID& id) {
    ops_.push_back(PendingOp{true, std::move(vault), Vector{}, Payload{}, id});
}

void Transaction::commit() {
    for (auto& op : ops_) {
        Vault& vault = db_->vault(op.vault);
        if (op.is_erase) {
            vault.erase(*op.id);
        } else {
            vault.place(op.vector, op.payload, op.id);
        }
    }
    ops_.clear();
    done_ = true;
}

RecordID TransactionVault::place(const Vector& vector, Payload payload,
                                 std::optional<RecordID> id) {
    const RecordID assigned = id.value_or(RecordID::generate());
    txn_->enqueue_place(vault_, vector, std::move(payload), assigned);
    return assigned;
}

void TransactionVault::erase(const RecordID& id) {
    txn_->enqueue_erase(vault_, id);
}

#ifdef ELIPS_GPU_ENABLED
gpu::GpuDeviceInfo ElipsInstance::gpu_info() const { return gpu_info_; }

gpu::GpuMetricsSnapshot ElipsInstance::gpu_stats() const { return gpu_stats_; }
#endif

}  // namespace elips
