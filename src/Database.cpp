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
#include "elips/quant_engine/Quantizer.hpp"
#include "elips/storage/FileSync.hpp"
#include "elips/storage/Serialization.hpp"
#include "elips/storage/WAL.hpp"
#include "elips/vector_engine/Metrics.hpp"

#ifdef ELIPS_GPU_ENABLED
#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/gpu_engine/GpuDeviceInfo.hpp"
#include "elips/gpu_engine/GpuMetricsSnapshot.hpp"
#include "elips/gpu_engine/BackendSelector.hpp"
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
// On-disk format version, shared by the snapshot, segments, the manifest, and
// the text-embedder manifest.
//
//   1 -> records without document/chunk/lineage extras
//   2 -> extras added
//   3 -> per-record codec byte and per-collection codebook section (quantization)
//
// Readers accept anything at or below this and reject only what is newer, so a
// v2 database opens unchanged under a v3 binary. Version comparisons in readers
// must be against *absolute* numbers, never against this constant: a gate
// written as `version >= snapshot_version` silently changes meaning the next
// time the constant moves.
constexpr std::uint32_t snapshot_version = 3U;
constexpr std::uint32_t version_with_extras = 2U;
constexpr std::uint32_t version_with_codecs = 3U;
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
    std::string collection_name;
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
    const std::set<std::string> query_set(query_terms.begin(), query_terms.end());
    const std::set<std::string> document_set(document_terms.begin(),
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
    return SearchResult{record.id,     distance_value, record.payload,
                        record.document, record.chunk, record.lineage,
                        record.codec};
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
        return fs::path(persisted.info.storage_path).make_preferred().string();
    }
    return (root / fs::path(persisted.info.storage_path)).make_preferred().string();
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

    detail::durable_rename(tmp, path / text_embedder_file);
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
    if (version > snapshot_version) {
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
        auto options = effective.local_text_embedder_options().value();
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

    const LocalTextEmbedderOptions options{
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

// Writes one record. From v3 a codec byte precedes the vector payload and
// selects which of the two forms follows: raw fp32 components, or a
// length-prefixed code. A compressed record is written as its code, never as a
// decoded approximation, so a checkpoint/reopen cycle is byte-idempotent.
void put_record(std::ostream& out, const Record& record, bool with_extras = true,
                bool with_codec = true) {
    out.write(reinterpret_cast<const char*>(record.id.bytes().data()),
              static_cast<std::streamsize>(record.id.bytes().size()));

    if (with_codec) {
        put<std::uint8_t>(out, static_cast<std::uint8_t>(record.codec));
    }
    if (with_codec && record.codec != quant::CodecId::none) {
        put<std::uint16_t>(out, static_cast<std::uint16_t>(record.codes.size()));
        out.write(reinterpret_cast<const char*>(record.codes.data()),
                  static_cast<std::streamsize>(record.codes.size()));
    } else {
        const auto values = record.vector.values();
        put<std::uint16_t>(out, static_cast<std::uint16_t>(values.size()));
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size_bytes()));
    }

    put_payload(out, record.payload);
    if (!with_extras) {
        return;
    }
    detail::put_document_attachment(out, record.document);
    detail::put_chunk_info(out, record.chunk);
    detail::put_embedding_lineage(out, record.lineage);
}

Record get_record(std::istream& in, bool with_extras, bool with_codec) {
    RecordID::Bytes bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));

    auto codec = quant::CodecId::none;
    if (with_codec) {
        const auto raw = get<std::uint8_t>(in);
        if (raw > static_cast<std::uint8_t>(quant::CodecId::sq8)) {
            throw StorageError{"unknown record codec tag"};
        }
        codec = static_cast<quant::CodecId>(raw);
    }

    std::vector<float> values;
    std::vector<std::uint8_t> codes;
    if (codec != quant::CodecId::none) {
        const auto code_len = get<std::uint16_t>(in);
        // Bound the prefix against the remaining stream before allocating for
        // it; see detail::check_length.
        detail::check_length(in, code_len);
        codes.resize(code_len);
        in.read(reinterpret_cast<char*>(codes.data()),
                static_cast<std::streamsize>(code_len));
    } else {
        const auto dim = get<std::uint16_t>(in);
        detail::check_length(in,
                             static_cast<std::uint64_t>(dim) * sizeof(float));
        values.resize(dim);
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(static_cast<std::size_t>(dim) * sizeof(float)));
    }
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

    return Record{RecordID{bytes},   Vector{std::move(values)},
                  std::move(payload), std::move(document),
                  std::move(chunk),   std::move(lineage),
                  std::move(codes),   codec};
}

// Per-collection codebook section, written after the collection name in both the
// monolithic snapshot and each segment. Absent (a zero byte) for an unquantized
// collection.
void put_codebook(std::ostream& out, const quant::QuantizerPtr& quantizer) {
    if (quantizer == nullptr) {
        put<std::uint8_t>(out, static_cast<std::uint8_t>(quant::CodecId::none));
        return;
    }
    put<std::uint8_t>(out, static_cast<std::uint8_t>(quantizer->codec()));
    std::ostringstream blob(std::ios::binary);
    quantizer->serialize(blob);
    const std::string bytes = blob.str();
    put<std::uint32_t>(out, static_cast<std::uint32_t>(bytes.size()));
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

quant::QuantizerPtr get_codebook(std::istream& in) {
    const auto raw = get<std::uint8_t>(in);
    if (raw == static_cast<std::uint8_t>(quant::CodecId::none)) {
        return {};
    }
    if (raw > static_cast<std::uint8_t>(quant::CodecId::sq8)) {
        throw StorageError{"unknown codebook codec tag"};
    }
    const auto length = get<std::uint32_t>(in);
    detail::check_length(in, length);
    std::string bytes(length, '\0');
    in.read(bytes.data(), static_cast<std::streamsize>(length));
    if (!in) {
        throw StorageError{"truncated codebook section"};
    }
    std::istringstream blob(bytes, std::ios::binary);
    return quant::load(blob);
}

void write_identity(const fs::path& file, const Config& config) {
    {
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
    detail::sync_file_path(file);
    detail::sync_directory(file.parent_path());
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
    const std::map<std::string, std::unique_ptr<Collection>>& collections) {
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
        put<std::uint32_t>(out, static_cast<std::uint32_t>(collections.size()));
        for (const auto& [name, collection] : collections) {
            put_string(out, name);
            // Codebook precedes the records: a reader has to install it before
            // it can interpret the codes that follow.
            put_codebook(out, collection->config().quantizer());
            const auto& records = collection->records();
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

    detail::durable_rename(tmp, path / snapshot_file);
}

void write_segment_file(const fs::path& file, const std::string& collection_name,
                        const Collection& collection) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw StorageError{"cannot write collection segment"};
    }
    put<std::uint32_t>(out, segment_magic);
    put<std::uint32_t>(out, snapshot_version);
    put_string(out, collection_name);
    put_codebook(out, collection.config().quantizer());
    const auto& records = collection.records();
    put<std::uint32_t>(out, static_cast<std::uint32_t>(records.size()));
    for (const auto& [id, record] : records) {
        (void)id;
        put_record(out, record);
    }
    if (!out) {
        throw StorageError{"error while writing collection segment"};
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
            put_string(out, entry.collection_name);
            put_string(out, entry.file_name);
        }
        if (!out) {
            throw StorageError{"error while writing manifest"};
        }
    }

    detail::durable_rename(tmp, path / manifest_file);
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
    if (version > snapshot_version) {
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

void load_snapshot_file(const fs::path& file, Database& instance) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw StorageError{"cannot open snapshot for reading"};
    }
    if (get<std::uint32_t>(in) != snapshot_magic) {
        throw StorageError{"snapshot magic mismatch"};
    }
    const auto version = get<std::uint32_t>(in);
    if (version > snapshot_version) {
        throw StorageError{"unsupported snapshot version"};
    }
    (void)get<std::uint16_t>(in);
    (void)get<std::uint8_t>(in);

    const auto collection_count = get<std::uint32_t>(in);
    for (std::uint32_t collection_index = 0; collection_index < collection_count;
         ++collection_index) {
        const std::string name = get_string(in);
        Collection& collection = instance.collection(name);
        collection.set_read_only(false);
        if (version >= version_with_codecs) {
            collection.install_quantizer(get_codebook(in));
        }
        const auto record_count = get<std::uint32_t>(in);
        for (std::uint32_t record_index = 0; record_index < record_count;
             ++record_index) {
            // Absolute version comparisons: a gate written against
            // snapshot_version would change meaning on the next bump and
            // silently mis-parse every older record.
            const Record record = get_record(in, version >= version_with_extras,
                                             version >= version_with_codecs);
            collection.place_record(record);
        }
    }
}

void load_segmented_state(const fs::path& root, Database& instance) {
    const auto entries = read_manifest_file(root / manifest_file);
    for (const auto& entry : entries) {
        const fs::path segment_path = root / segment_dir / entry.file_name;
        std::ifstream in(segment_path, std::ios::binary);
        if (!in) {
            throw StorageError{"missing collection segment: " + entry.file_name};
        }
        if (get<std::uint32_t>(in) != segment_magic) {
            throw StorageError{"segment magic mismatch"};
        }
        const auto version = get<std::uint32_t>(in);
        if (version > snapshot_version) {
            throw StorageError{"unsupported segment version"};
        }
        const std::string stored_name = get_string(in);
        if (stored_name != entry.collection_name) {
            throw StorageError{"segment collection mismatch for " + entry.file_name};
        }
        Collection& collection = instance.collection(entry.collection_name);
        collection.set_read_only(false);
        if (version >= version_with_codecs) {
            collection.install_quantizer(get_codebook(in));
        }
        const auto record_count = get<std::uint32_t>(in);
        for (std::uint32_t record_index = 0; record_index < record_count;
             ++record_index) {
            const Record record =
                get_record(in, true, version >= version_with_codecs);
            collection.place_record(record);
        }
    }
}

void configure_gpu_backend(Database& instance, const Config& config) {
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

    gpu::BackendSelector selector;
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

void apply_read_only_mode(Database& instance) {
    for (const auto& name : instance.list_collections()) {
        instance.collection(name).set_read_only(true);
    }
}

}  // namespace

// --------------------------------- Collection ---------------------------------

Collection::Collection(std::string name, const Config& config
#ifdef ELIPS_GPU_ENABLED
             ,
             gpu::GpuBackend* gpu_backend
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

Vector Collection::prepare(const Vector& vector) const {
    if (vector.dimension() != config_.dimension()) {
        throw DimensionMismatch{"vector dimension does not match collection: "
                                "expected d=" +
                                std::to_string(config_.dimension()) + ", got " +
                                std::to_string(vector.dimension())};
    }
    if (!all_finite(vector.values())) {
        throw InvalidVector{"vector contains NaN or Inf"};
    }
    return requires_normalization(config_.metric()) ? vector.normalized()
                                                    : vector;
}

void Collection::ensure_writable() const {
    if (sealed_) {
        throw StorageError{
            "database is closed: this write would never be persisted"};
    }
    if (read_only_) {
        throw StorageError{"collection is opened in read-only mode"};
    }
}

void Collection::set_read_only(bool read_only) noexcept {
    const std::unique_lock lock(mutex_);
    read_only_ = read_only;
}

bool Collection::read_only() const noexcept {
    const std::shared_lock lock(mutex_);
    return read_only_;
}

void Collection::seal() noexcept {
    const std::unique_lock lock(mutex_);
    sealed_ = true;
}

bool Collection::sealed() const noexcept {
    const std::shared_lock lock(mutex_);
    return sealed_;
}

std::map<RecordID, Record> Collection::records() const {
    const std::shared_lock lock(mutex_);
    return records_;
}

RecordID Collection::add(const Vector& vector, Payload payload,
                      std::optional<RecordID> id,
                      std::optional<DocumentAttachment> document,
                      std::optional<ChunkInfo> chunk,
                      std::optional<EmbeddingLineage> lineage) {
    const std::unique_lock lock(mutex_);
    return place_locked(vector, std::move(payload), id,
                        std::move(document), std::move(chunk),
                        std::move(lineage));
}

RecordID Collection::place_locked(const Vector& vector, Payload payload,
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

    index_->add_with_ids(record_id, prepared.values());
    if (config_.metadata_acceleration()) {
        metadata_index_.insert(record_id, payload);
    }

    Record stored{record_id,
                  std::move(prepared),
                  std::move(payload),
                  std::move(document),
                  std::move(chunk),
                  std::move(lineage),
                  {},
                  quant::CodecId::none};

    // Once a codebook exists, the code is what the collection keeps: the fp32 vector
    // is dropped rather than stored alongside, which is where the memory saving
    // comes from.
    if (const auto& codec = quantizer(); codec != nullptr) {
        stored.codes = codec->encode(stored.vector.values());
        stored.codec = codec->codec();
        stored.vector = Vector{};
    }

    records_[record_id] = std::move(stored);
    return record_id;
}

Vector Collection::reconstruct(const Record& record) const {
    if (record.codes.empty()) {
        return record.vector;
    }
    const auto& codec = quantizer();
    if (codec == nullptr) {
        // A code with no quantizer to read it means the codebook failed to load.
        // Returning an empty vector would look like a legitimately empty record.
        throw StorageError{
            "record is quantized but the collection has no matching codebook"};
    }
    return Vector{codec->decode(record.codes)};
}

Record Collection::hydrate(const Record& record) const {
    if (record.codes.empty()) {
        return record;
    }
    Record out = record;
    out.vector = reconstruct(record);
    return out;
}

void Collection::place_record(const Record& record) {
    const std::unique_lock lock(mutex_);
    place_record_locked(record);
}

void Collection::place_record_locked(const Record& record) {
    if (record.codes.empty()) {
        place_locked(record.vector, record.payload, record.id, record.document,
                     record.chunk, record.lineage);
        return;
    }

    ensure_writable();

    // A record that arrives already coded is stored byte-for-byte. It is not
    // normalized (it was normalized before it was first encoded) and not
    // re-encoded, so a checkpoint/reopen cycle is idempotent no matter how many
    // times it runs.
    const auto& codec = quantizer();
    if (codec == nullptr || record.codes.size() != codec->code_bytes()) {
        throw StorageError{"quantized record does not match the collection codebook"};
    }

    if (wal_ != nullptr) {
        wal_->append_insert_quantized(name_, record.id, record.codes,
                                      record.codec, record.payload,
                                      record.document, record.chunk,
                                      record.lineage);
    }

    const auto existing = records_.find(record.id);
    if (existing != records_.end()) {
        if (config_.metadata_acceleration()) {
            metadata_index_.remove(record.id, existing->second.payload);
        }
        index_->remove(record.id);
    }

    // The index takes vectors, so the code is decoded once for navigation. It is
    // re-encoded internally to the identical code, since encoding is
    // deterministic against a fixed codebook.
    const Vector decoded = reconstruct(record);
    index_->add_with_ids(record.id, decoded.values());
    if (config_.metadata_acceleration()) {
        metadata_index_.insert(record.id, record.payload);
    }

    Record stored = record;
    stored.vector = Vector{};
    records_[record.id] = std::move(stored);
}

RecordID Collection::add_document(std::string text, Payload payload,
                               std::optional<RecordID> id,
                               std::optional<ChunkInfo> chunk,
                               std::optional<EmbeddingLineage> lineage) {
    if (!config_.has_text_embedder()) {
        throw ConfigError{missing_text_embedder_message(config_)};
    }

    const RecordID record_id = id.value_or(RecordID::generate());
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

    // Embed outside the collection lock: it is a pure function of the text and can
    // be slow, so holding the exclusive lock across it would serialize readers
    // for no reason.
    const Vector embedded = config_.text_embedder()->embed(text);
    DocumentAttachment document{
        .text = std::move(text),
        .uri = {},
        .mime_type = "text/plain",
    };
    return add(embedded, std::move(payload), record_id, std::move(document),
                 std::move(chunk), std::move(lineage));
}

void Collection::add_many(const std::vector<Record>& records) {
    const std::unique_lock lock(mutex_);
    // Check up front so an empty batch against a sealed or read-only collection
    // still reports the problem, rather than silently succeeding.
    ensure_writable();
    for (const auto& record : records) {
        const std::optional<RecordID> id =
            (record.id == RecordID{}) ? std::nullopt
                                      : std::optional<RecordID>{record.id};
        place_locked(record.vector, record.payload, id, record.document,
                     record.chunk, record.lineage);
    }
}

std::vector<SearchResult> Collection::search_records(
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

    // Compressed collections score by asymmetric lookup: one table built here, then
    // a gather per record. Cheaper than the fp32 scan it replaces, and it avoids
    // decoding every candidate just to measure it.
    const auto& codec = quantizer();
    std::vector<float> lut;
    if (codec != nullptr) {
        lut = codec->make_lut(prepared.values());
    }

    const auto accumulate = [&](const Record& record) {
        if (!filter.matches(record.payload)) {
            return;
        }
        const float distance_value =
            (codec != nullptr && !record.codes.empty())
                ? codec->lut_distance(lut, record.codes)
                : distance(config_.metric(), prepared.values(),
                           record.vector.values());
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

QueryPlan Collection::plan_seek(const Vector& prepared, std::size_t top,
                           const Filter& filter,
                           std::optional<float> threshold,
                           bool has_text_component) const {
    (void)prepared;

    QueryPlan plan;
    plan.index_type = std::string(index_->type_name());
    plan.gpu_index = begins_with(plan.index_type, "gpu_");
    plan.candidate_count = records_.size();
    if (const auto& codec = quantizer(); codec != nullptr) {
        plan.codec = codec->codec();
    }

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

std::vector<SearchResult> Collection::search(const Vector& query, std::size_t top,
                                      const Filter& filter,
                                      std::optional<float> threshold) const {
    const std::shared_lock lock(mutex_);
    return seek_locked(query, top, filter, threshold);
}

std::vector<SearchResult> Collection::seek_locked(
    const Vector& query, std::size_t top, const Filter& filter,
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

    // Post-filtering discards hits, so a fixed over-fetch can return fewer than
    // `top`. Re-probe with a wider beam until the filter is satisfied or the
    // whole collection has been swept. Bounded by records_.size(), so worst case
    // degenerates to a full scan rather than an unbounded loop.
    std::vector<SearchResult> results;
    while (true) {
        const auto hits = index_->search(prepared.values(), fetch);
        results.clear();
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
        const bool satisfied = results.size() >= top;
        const bool exhausted = fetch >= records_.size() || hits.size() < fetch;
        if (satisfied || exhausted) {
            return results;
        }
        fetch = std::min(records_.size(), fetch * 4);
    }
}

std::vector<SearchResult> Collection::search_text(std::string_view text, std::size_t top,
                                           const Filter& filter,
                                           std::optional<float> threshold) const {
    if (!config_.has_text_embedder()) {
        throw ConfigError{missing_text_embedder_message(config_)};
    }
    if (top == 0) {
        return {};
    }
    // Embed before taking the lock; embedding does not touch collection state.
    const Vector embedded = config_.text_embedder()->embed(text);
    const std::shared_lock lock(mutex_);
    return seek_locked(embedded, top, filter, threshold);
}

std::vector<SearchResult> Collection::hybrid_search(const Vector& query,
                                             std::string_view text,
                                             std::size_t top,
                                             const Filter& filter,
                                             std::optional<float> threshold,
                                             float lexical_weight) const {
    const std::shared_lock lock(mutex_);
    if (top == 0 || records_.empty()) {
        return {};
    }

    const float weight = std::clamp(lexical_weight, 0.0F, 1.0F);
    if (text.empty() || weight == 0.0F) {
        return seek_locked(query, top, filter, threshold);
    }

    const std::size_t candidate_top =
        std::min(records_.size(), std::max<std::size_t>(top * 5, top));
    auto candidates = seek_locked(query, candidate_top, filter, threshold);
    for (auto& result : candidates) {
        const auto it = records_.find(result.id);
        const float lexical_score =
            (it != records_.end() && it->second.document.has_value())
                ? lexical_overlap_score(text, it->second.document.value().text)
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

QueryPlan Collection::explain_query(const Vector& query, std::size_t top,
                              const Filter& filter,
                              std::optional<float> threshold,
                              bool has_text_component) const {
    const std::shared_lock lock(mutex_);
    return plan_seek(prepare(query), top, filter, threshold, has_text_component);
}

std::vector<Record> Collection::scan(const Filter& filter, std::size_t offset,
                                std::size_t limit) const {
    const std::shared_lock lock(mutex_);
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
        out.push_back(hydrate(record));
    }
    return out;
}

std::optional<Record> Collection::get(const RecordID& id) const {
    const std::shared_lock lock(mutex_);
    const auto it = records_.find(id);
    if (it == records_.end()) {
        return std::nullopt;
    }
    return hydrate(it->second);
}

bool Collection::remove(const RecordID& id) {
    const std::unique_lock lock(mutex_);
    return erase_locked(id);
}

bool Collection::erase_locked(const RecordID& id) {
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

void Collection::restore_for_undo(const RecordID& id,
                             const std::optional<Record>& previous) {
    const std::unique_lock lock(mutex_);
    const auto existing = records_.find(id);
    if (existing != records_.end()) {
        if (config_.metadata_acceleration()) {
            metadata_index_.remove(id, existing->second.payload);
        }
        index_->remove(id);
        records_.erase(existing);
    }
    if (!previous.has_value()) {
        return;
    }
    // `previous` came from fetch(), so a compressed record carries both its
    // original code and a reconstruction. Restoring the code verbatim is what
    // keeps rollback lossless: re-encoding the reconstruction would cost the
    // record a generation of accuracy it never asked for, on a transaction that
    // may not even have modified it.
    const Vector vector = reconstruct(*previous);
    index_->add_with_ids(id, vector.values());
    if (config_.metadata_acceleration()) {
        metadata_index_.insert(id, previous->payload);
    }
    Record restored = *previous;
    if (!restored.codes.empty()) {
        restored.vector = Vector{};
    }
    records_[id] = std::move(restored);
}

void Collection::rebuild_index() {
    const std::unique_lock lock(mutex_);
    rebuild_index_locked();
}

void Collection::rebuild_index_locked() {
    ensure_writable();

    auto rebuilt = make_index(config_, config_.dimension()
#ifdef ELIPS_GPU_ENABLED
                              ,
                              gpu_backend_
#endif
    );

    metadata_index_ = MetadataFilterIndex{};
    for (const auto& [id, record] : records_) {
        // Compressed records decode once here; the index re-encodes internally
        // to the same code, so a rebuild does not degrade what is stored.
        const Vector vector = reconstruct(record);
        rebuilt->add_with_ids(id, vector.values());
        if (config_.metadata_acceleration()) {
            metadata_index_.insert(id, record.payload);
        }
    }
    index_ = std::move(rebuilt);
}

bool Collection::is_trained() const noexcept {
    const std::shared_lock lock(mutex_);
    return quantizer() != nullptr;
}

quant::CodecId Collection::codec() const noexcept {
    const std::shared_lock lock(mutex_);
    const auto& codec = quantizer();
    return codec != nullptr ? codec->codec() : quant::CodecId::none;
}

void Collection::install_quantizer(quant::QuantizerPtr quantizer) {
    const std::unique_lock lock(mutex_);
    if (quantizer != nullptr && quantizer->dimension() != config_.dimension()) {
        throw StorageError{"persisted codebook dimension does not match the collection"};
    }
    config_.attach_quantizer(std::move(quantizer));
    // The index was built by the constructor without a codebook, so it is
    // storing fp32. Rebuild it so it stores codes -- but only if records are
    // already present; during load this runs before any placement and the
    // rebuild is a no-op, with make_index() picking the codebook up naturally.
    if (!records_.empty()) {
        rebuild_index_locked();
    } else {
        index_ = make_index(config_, config_.dimension()
#ifdef ELIPS_GPU_ENABLED
                            ,
                            gpu_backend_
#endif
        );
    }
}

void Collection::train() {
    const std::unique_lock lock(mutex_);
    ensure_writable();

    if (!config_.has_quantization()) {
        throw ConfigError{
            "collection has no quantization codec configured; set one via "
            "Config::quantization() before calling quantize()"};
    }
    if (quantizer() != nullptr) {
        throw ConfigError{"collection is already quantized"};
    }
    if (records_.empty()) {
        throw ConfigError{
            "cannot train a codebook on an empty collection: quantization needs "
            "representative data"};
    }

    const auto dimension = config_.dimension();
    quant::validate(config_.quantization(), dimension);

    // Train on a bounded sample. Codebook quality is a function of how well the
    // sample covers the distribution, not of its size, so this keeps training
    // cost tied to the dimension and codebook width rather than to the collection.
    const std::size_t sample_size =
        std::min(records_.size(), quant::train_sample_cap);
    const std::size_t stride = std::max<std::size_t>(1, records_.size() / sample_size);

    std::vector<float> training;
    training.reserve(sample_size * dimension);
    std::size_t taken = 0;
    std::size_t position = 0;
    for (const auto& [id, record] : records_) {
        (void)id;
        if (position++ % stride != 0 || taken >= sample_size) {
            continue;
        }
        const auto values = record.vector.values();
        training.insert(training.end(), values.begin(), values.end());
        ++taken;
    }

    auto trained = quant::train(config_.quantization(), config_.metric(),
                                dimension, training, taken);

    // Encode every record, then drop the fp32 copies. Done before the index is
    // rebuilt so the rebuild reads codes through reconstruct() exactly as every
    // later rebuild will.
    for (auto& [id, record] : records_) {
        (void)id;
        record.codes = trained->encode(record.vector.values());
        record.codec = trained->codec();
        record.vector = Vector{};
    }

    config_.attach_quantizer(std::move(trained));
    rebuild_index_locked();
}

void Database::train(const std::string& collection_name) {
    const std::lock_guard lock(mutex_);
    if (closed_) {
        throw StorageError{"database is closed"};
    }
    collection_locked(collection_name).train();

    // Order matters and is the reason no partial-quantize state can exist on
    // disk: the checkpoint durably publishes both the codebook and the encoded
    // records, and only then is the log truncated. A crash before the checkpoint
    // leaves the previous fp32 database intact; a crash after it leaves a fully
    // quantized one. There is no window where the log holds insert_q records
    // whose codebook was never written.
    if (persistent_) {
        checkpoint_locked();
    }
}

void Database::quantize_all() {
    for (const auto& name : list_collections()) {
        train(name);
    }
}

void Collection::vacuum() {
    const std::unique_lock lock(mutex_);
    ensure_writable();
    index_->vacuum();
}

std::size_t Collection::pending_removals() const noexcept {
    const std::shared_lock lock(mutex_);
    return index_->pending_removals();
}

CollectionInfo Collection::info() const noexcept {
    const std::shared_lock lock(mutex_);
    const auto& codec = quantizer();
    return CollectionInfo{records_.size(), config_.dimension(), config_.metric(),
                     codec != nullptr ? codec->codec() : quant::CodecId::none,
                     codec != nullptr ? codec->code_bytes() : 0};
}

// ----------------------------- Database -----------------------------

Database::Database(std::string path, Config config, bool persistent,
                             std::optional<LockManager> lock)
    : path_(std::move(path)),
      config_(std::move(config)),
      persistent_(persistent),
      lock_(std::move(lock)) {}

Database::~Database() {
    const std::lock_guard lock(mutex_);
    if (persistent_ && !closed_ && config_.access_mode() != AccessMode::read_only) {
        try {
            checkpoint_locked();
        } catch (...) {
            // E.16: destructors must not throw. Best-effort checkpoint.
            (void)0;
        }
    }

    // Collection-owned GPU indexes hold non-owning backend references, so release
    // every collection before member destruction reaches gpu_backend_.
    collections_.clear();
}

void Database::abandon() noexcept {
    const std::lock_guard lock(mutex_);
    closed_ = true;
}

Collection& Database::collection(const std::string& name) {
    const std::lock_guard lock(mutex_);
    return collection_locked(name);
}

Collection& Database::collection_locked(const std::string& name) {
    const auto it = collections_.find(name);
    if (it != collections_.end()) {
        return *it->second;
    }

#ifdef ELIPS_GPU_ENABLED
    auto created = std::make_unique<Collection>(name, config_, gpu_backend_.get());
#else
    auto created = std::make_unique<Collection>(name, config_);
#endif
    created->set_wal(wal_.get());
    if (config_.access_mode() == AccessMode::read_only) {
        created->set_read_only(true);
    }
    if (closed_) {
        created->seal();
    }

    Collection& ref = *created;
    collections_.emplace(name, std::move(created));
    return ref;
}

Collection& Database::adopt_collection(std::unique_ptr<Collection> collection) {
    const std::lock_guard lock(mutex_);
    collection->set_wal(wal_.get());
    if (config_.access_mode() == AccessMode::read_only) {
        collection->set_read_only(true);
    }
    Collection& ref = *collection;
    collections_[collection->name()] = std::move(collection);
    return ref;
}

void Database::attach_wal(std::unique_ptr<WAL> wal) {
    const std::lock_guard lock(mutex_);
    wal_ = std::move(wal);
    for (auto& [name, collection] : collections_) {
        (void)name;
        collection->set_wal(wal_.get());
    }
}

std::vector<std::string> Database::list_collections() const {
    const std::lock_guard lock(mutex_);
    std::vector<std::string> names;
    names.reserve(collections_.size());
    for (const auto& [name, collection] : collections_) {
        (void)collection;
        names.push_back(name);
    }
    return names;
}

void Database::checkpoint() {
    const std::lock_guard lock(mutex_);
    checkpoint_locked();
}

void Database::checkpoint_locked() {
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
        entries.reserve(collections_.size());

        std::size_t ordinal = 0;
        for (const auto& [name, collection] : collections_) {
            const std::string file_name =
                "vault_" + std::to_string(ordinal++) + "_" +
                std::to_string(epoch) + ".segment";
            const fs::path tmp_path = segments_root / (file_name + ".tmp");
            const fs::path final_path = segments_root / file_name;
            write_segment_file(tmp_path, name, *collection);
            detail::durable_rename(tmp_path, final_path);
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
        write_snapshot_file(root, config_, collections_);
        std::error_code ec;
        fs::remove(root / manifest_file, ec);
        fs::remove_all(root / segment_dir, ec);
    }

    if (wal_ != nullptr) {
        wal_->reset();
    }
}

void Database::compact() {
    const std::lock_guard lock(mutex_);
    if (!persistent_ || config_.access_mode() == AccessMode::read_only) {
        return;
    }
    for (auto& [name, collection] : collections_) {
        (void)name;
        collection->rebuild_index();
    }
    checkpoint_locked();
}

void Database::vacuum() {
    const std::lock_guard lock(mutex_);
    if (config_.access_mode() == AccessMode::read_only) {
        return;
    }
    for (auto& [name, collection] : collections_) {
        (void)name;
        collection->vacuum();
    }
}

void Database::close() {
    const std::lock_guard lock(mutex_);
    if (closed_) {
        return;
    }

    checkpoint_locked();
    for (auto& [name, collection] : collections_) {
        (void)name;
        collection->set_wal(nullptr);
        // A write after close() would never be WAL-logged and never
        // checkpointed: it would return success and then vanish. Refuse it.
        collection->seal();
    }
    wal_.reset();
    lock_.reset();
    closed_ = true;
}

// --------------------------------- open ----------------------------------

std::unique_ptr<Database> open(const std::string& path,
                                    const Config& config) {
    if (path == ":memory:") {
        if (config.dimension() == 0) {
            throw ConfigError{"in-memory database requires a dimension"};
        }

        Config effective = config;
        (void)resolve_text_embedder_for_open(fs::path{}, /*persistent=*/false,
                                             /*new_database=*/true, effective);
        auto instance =
            std::make_unique<Database>(path, effective, /*persistent=*/false);
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

    auto instance = std::make_unique<Database>(path, effective,
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
        Collection& collection = instance->collection(entry.collection);
        collection.set_read_only(false);
        if (entry.op == WAL::Op::insert_q) {
            // The codebook came from the snapshot that was loaded above.
            // quantize() checkpoints before resetting the log, so an insert_q
            // record can never predate the codebook that decodes it.
            collection.place_record(Record{entry.id, Vector{}, entry.payload,
                                      entry.document, entry.chunk, entry.lineage,
                                      entry.codes, entry.codec});
        } else if (entry.op == WAL::Op::insert || entry.op == WAL::Op::insert_ex) {
            collection.add(Vector{entry.vector}, entry.payload, entry.id, entry.document,
                        entry.chunk, entry.lineage);
        } else {
            collection.remove(entry.id);
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

Transaction Database::begin_transaction() { return Transaction{*this}; }

Transaction::~Transaction() {
    if (!done_) {
        rollback();
    }
}

void Transaction::enqueue_place(std::string collection, const Vector& vector,
                                Payload payload,
                                std::optional<RecordID> id) {
    if (vector.dimension() != db_->config().dimension()) {
        throw DimensionMismatch{"vector dimension does not match database: "
                                "expected d=" +
                                std::to_string(db_->config().dimension()) +
                                ", got " + std::to_string(vector.dimension())};
    }
    if (!all_finite(vector.values())) {
        throw InvalidVector{"vector contains NaN or Inf"};
    }
    ops_.push_back(PendingOp{false, std::move(collection), vector, std::move(payload),
                             id});
}

void Transaction::enqueue_erase(std::string collection, const RecordID& id) {
    ops_.push_back(PendingOp{true, std::move(collection), Vector{}, Payload{}, id});
}

void Transaction::undo(const std::vector<UndoEntry>& entries) noexcept {
    // Reverse order so repeated writes to the same id land on the oldest state.
    for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        try {
            db_->collection_locked(it->collection).restore_for_undo(it->id, it->previous);
        } catch (...) {
            // Undo is a best-effort in-memory operation on structures that were
            // just successfully mutated; nothing here can be usefully retried.
            (void)0;
        }
    }
}

void Transaction::commit() {
    // Hold the instance lock for the whole batch so a concurrent writer cannot
    // interleave its own mutations between our ops (which would make the undo
    // log restore state that is no longer the correct pre-batch state).
    const std::lock_guard db_lock(db_->mutex_);

    // Eager checks that cover every failure mode we can see before mutating:
    // collection existence and writability. Runtime I/O failure is still possible,
    // which is what the undo log below is for.
    for (const auto& op : ops_) {
        const Collection& collection = db_->collection_locked(op.collection);
        if (collection.read_only() || collection.sealed()) {
            throw StorageError{collection.sealed()
                                   ? "database is closed: this write would "
                                     "never be persisted"
                                   : "collection is opened in read-only mode"};
        }
    }

    WAL* wal = db_->wal_.get();
    if (wal != nullptr) {
        wal->append_txn_begin();
    }

    std::vector<UndoEntry> undo_log;
    undo_log.reserve(ops_.size());
    try {
        for (auto& op : ops_) {
            Collection& collection = db_->collection_locked(op.collection);
            const RecordID target =
                op.id.value_or(op.is_erase ? RecordID{} : RecordID::generate());
            undo_log.push_back(
                UndoEntry{op.collection, target, collection.get(target)});
            if (op.is_erase) {
                collection.remove(target);
            } else {
                collection.add(op.vector, op.payload, target);
            }
        }
    } catch (...) {
        undo(undo_log);
        // No commit marker is written, so replay discards the WAL records this
        // batch already appended.
        throw;
    }

    if (wal != nullptr) {
        wal->append_txn_commit();
    }
    ops_.clear();
    done_ = true;
}

RecordID TransactionCollection::add(const Vector& vector, Payload payload,
                                 std::optional<RecordID> id) {
    const RecordID assigned = id.value_or(RecordID::generate());
    txn_->enqueue_place(collection_, vector, std::move(payload), assigned);
    return assigned;
}

void TransactionCollection::remove(const RecordID& id) {
    txn_->enqueue_erase(collection_, id);
}

#ifdef ELIPS_GPU_ENABLED
gpu::GpuDeviceInfo Database::gpu_info() const { return gpu_info_; }

gpu::GpuMetricsSnapshot Database::gpu_stats() const { return gpu_stats_; }
#endif

}  // namespace elips
