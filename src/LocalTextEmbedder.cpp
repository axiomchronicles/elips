#include "elips/text_engine/LocalTextEmbedder.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>

#include "elips/domain/Errors.hpp"
#include "elips/storage/Serialization.hpp"

namespace elips {

struct LocalTextEmbedder::Artifact {
    std::uint16_t dimension{0};
    std::uint64_t seed{0};
    std::uint8_t min_char_ngram{0};
    std::uint8_t max_char_ngram{0};
    std::uint8_t projections_per_feature{0};
    bool include_bigrams{false};
};

namespace {

namespace fs = std::filesystem;

using detail::get;
using detail::get_string;
using detail::put;
using detail::put_string;

constexpr std::uint32_t artifact_magic = 0xE1105E21U;
constexpr std::uint32_t artifact_version = 1U;
constexpr std::string_view local_provider = "elips-local";
constexpr std::string_view local_backend = "builtin-hash";

struct ModelDescriptor {
    std::string_view model;
    std::string_view revision;
    std::uint64_t seed;
    std::uint8_t min_char_ngram;
    std::uint8_t max_char_ngram;
    std::uint8_t projections_per_feature;
    bool include_bigrams;
};

[[nodiscard]] std::string hex64(const std::uint64_t value) {
    std::ostringstream out;
    out << std::hex;
    out.width(16);
    out.fill('0');
    out << value;
    return out.str();
}

[[nodiscard]] std::uint64_t fnv1a(std::string_view text) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : text) {
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] const ModelDescriptor& resolve_model(std::string_view model,
                                                   std::string_view revision) {
    static const std::vector<ModelDescriptor> models{
        ModelDescriptor{
            .model = "default",
            .revision = "v1",
            .seed = 0x6A09E667F3BCC909ULL,
            .min_char_ngram = 3U,
            .max_char_ngram = 5U,
            .projections_per_feature = 4U,
            .include_bigrams = true,
        },
        ModelDescriptor{
            .model = "compact",
            .revision = "v1",
            .seed = 0xBB67AE8584CAA73BULL,
            .min_char_ngram = 3U,
            .max_char_ngram = 4U,
            .projections_per_feature = 2U,
            .include_bigrams = true,
        },
    };

    const auto it = std::find_if(models.begin(), models.end(),
                                 [model, revision](const ModelDescriptor& entry) {
                                     return entry.model == model &&
                                            entry.revision == revision;
                                 });
    if (it != models.end()) {
        return *it;
    }

    throw ConfigError{"unknown local text embedder model '" + std::string(model) +
                      "@" + std::string(revision) + "'"};
}

[[nodiscard]] std::string compute_fingerprint(const ModelDescriptor& descriptor,
                                              const std::uint16_t dimension) {
    std::ostringstream key;
    key << local_provider << '|'
        << descriptor.model << '|'
        << descriptor.revision << '|'
        << local_backend << '|'
        << dimension << '|'
        << descriptor.seed << '|'
        << static_cast<unsigned int>(descriptor.min_char_ngram) << '|'
        << static_cast<unsigned int>(descriptor.max_char_ngram) << '|'
        << static_cast<unsigned int>(descriptor.projections_per_feature) << '|'
        << (descriptor.include_bigrams ? '1' : '0');
    return hex64(fnv1a(key.str()));
}

[[nodiscard]] std::string default_storage_path(const LocalTextEmbedderOptions& options,
                                               const std::uint16_t dimension) {
    const fs::path root = fs::temp_directory_path() / "elips_text_embedders";
    fs::create_directories(root);
    return (root / (std::string(local_provider) + "_" + options.model + "_" +
                    options.revision + "_" + std::to_string(dimension) +
                    ".localembed"))
        .string();
}

[[nodiscard]] LocalTextEmbedderOptions resolve_options(
    LocalTextEmbedderOptions options, const std::uint16_t fallback_dimension,
    const std::string& preferred_storage_path = {}) {
    if (options.dimension == 0U) {
        options.dimension = fallback_dimension;
    }
    if (options.dimension == 0U) {
        throw ConfigError{"local text embedder requires a non-zero dimension"};
    }
    (void)resolve_model(options.model, options.revision);
    if (options.storage_path.empty()) {
        options.storage_path = preferred_storage_path.empty()
                                   ? default_storage_path(options, options.dimension)
                                   : preferred_storage_path;
    }
    return options;
}

void ensure_artifact_written(const LocalTextEmbedderOptions& options) {
    const auto& descriptor = resolve_model(options.model, options.revision);
    const fs::path path = options.storage_path;
    std::error_code ec;
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path(), ec);
    }
    if (ec) {
        throw StorageError{"cannot create local text embedder directory: " +
                           path.parent_path().string()};
    }
    if (fs::exists(path, ec) && !ec) {
        return;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw StorageError{"cannot write local text embedder artifact: " +
                           path.string()};
    }

    put<std::uint32_t>(out, artifact_magic);
    put<std::uint32_t>(out, artifact_version);
    put_string(out, std::string(local_provider));
    put_string(out, options.model);
    put_string(out, options.revision);
    put_string(out, std::string(local_backend));
    put<std::uint16_t>(out, options.dimension);
    put<std::uint64_t>(out, descriptor.seed);
    put<std::uint8_t>(out, descriptor.min_char_ngram);
    put<std::uint8_t>(out, descriptor.max_char_ngram);
    put<std::uint8_t>(out, descriptor.projections_per_feature);
    put<std::uint8_t>(out, descriptor.include_bigrams ? 1U : 0U);
    put_string(out, compute_fingerprint(descriptor, options.dimension));
    if (!out) {
        throw StorageError{"error while writing local text embedder artifact: " +
                           path.string()};
    }
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

std::vector<std::string> collect_features(std::string_view text,
                                          const std::uint8_t min_char_ngram,
                                          const std::uint8_t max_char_ngram,
                                          const bool include_bigrams) {
    auto tokens = tokenize_terms(text);
    std::vector<std::string> features;
    features.reserve(tokens.size() * 4U);

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        features.push_back(tokens[i]);
        if (include_bigrams && (i + 1U) < tokens.size()) {
            features.push_back(tokens[i] + "_" + tokens[i + 1U]);
        }

        const std::string bounded = "^" + tokens[i] + "$";
        for (std::uint8_t n = min_char_ngram; n <= max_char_ngram; ++n) {
            if (bounded.size() < n) {
                continue;
            }
            for (std::size_t start = 0; start + n <= bounded.size(); ++start) {
                features.push_back(bounded.substr(start, n));
            }
        }
    }

    return features;
}

}  // namespace

TextEmbedderInfo describe_local_text_embedder(const LocalTextEmbedderOptions& options,
                                              const std::uint16_t fallback_dimension,
                                              const bool auto_attached) {
    const auto& descriptor = resolve_model(options.model, options.revision);
    const std::uint16_t dimension =
        options.dimension == 0U ? fallback_dimension : options.dimension;

    return TextEmbedderInfo{
        .kind = TextEmbedderKind::local_builtin,
        .provider = std::string(local_provider),
        .model = options.model,
        .revision = options.revision,
        .backend = std::string(local_backend),
        .dimension = dimension,
        .fingerprint =
            dimension == 0U ? std::string{} : compute_fingerprint(descriptor, dimension),
        .storage_path = options.storage_path,
        .rehydratable = true,
        .loaded = false,
        .auto_attached = auto_attached,
    };
}

LocalTextEmbedder::LocalTextEmbedder(LocalTextEmbedderOptions options)
    : options_([&options] {
          const auto fallback_dimension = options.dimension;
          return resolve_options(std::move(options), fallback_dimension);
      }()),
      fingerprint_(
          compute_fingerprint(resolve_model(options_.model, options_.revision),
                              options_.dimension)) {
    ensure_artifact_written(options_);
}

LocalTextEmbedder::~LocalTextEmbedder() = default;

void LocalTextEmbedder::ensure_loaded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (artifact_ != nullptr) {
        return;
    }

    std::ifstream in(options_.storage_path, std::ios::binary);
    if (!in) {
        throw StorageError{"local text embedder artifact is missing: " +
                           options_.storage_path};
    }
    if (get<std::uint32_t>(in) != artifact_magic) {
        throw StorageError{"local text embedder artifact magic mismatch: " +
                           options_.storage_path};
    }
    if (get<std::uint32_t>(in) != artifact_version) {
        throw StorageError{
            "unsupported local text embedder artifact version: " +
            options_.storage_path};
    }

    const std::string provider = get_string(in);
    const std::string model = get_string(in);
    const std::string revision = get_string(in);
    const std::string backend = get_string(in);
    auto loaded = std::make_unique<Artifact>();
    loaded->dimension = get<std::uint16_t>(in);
    loaded->seed = get<std::uint64_t>(in);
    loaded->min_char_ngram = get<std::uint8_t>(in);
    loaded->max_char_ngram = get<std::uint8_t>(in);
    loaded->projections_per_feature = get<std::uint8_t>(in);
    loaded->include_bigrams = get<std::uint8_t>(in) != 0U;
    const std::string stored_fingerprint = get_string(in);
    if (!in) {
        throw StorageError{"truncated local text embedder artifact: " +
                           options_.storage_path};
    }

    if (provider != local_provider || backend != local_backend ||
        model != options_.model || revision != options_.revision ||
        loaded->dimension != options_.dimension) {
        throw StorageError{"local text embedder artifact metadata mismatch: " +
                           options_.storage_path};
    }

    const auto& descriptor = resolve_model(options_.model, options_.revision);
    if (loaded->seed != descriptor.seed ||
        loaded->min_char_ngram != descriptor.min_char_ngram ||
        loaded->max_char_ngram != descriptor.max_char_ngram ||
        loaded->projections_per_feature != descriptor.projections_per_feature ||
        loaded->include_bigrams != descriptor.include_bigrams) {
        throw StorageError{"local text embedder artifact parameters are corrupt: " +
                           options_.storage_path};
    }

    if (stored_fingerprint != fingerprint_) {
        throw StorageError{"local text embedder artifact fingerprint mismatch: " +
                           options_.storage_path};
    }

    artifact_ = std::move(loaded);
}

Vector LocalTextEmbedder::embed(std::string_view text) const {
    ensure_loaded();

    Artifact artifact;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        artifact = *artifact_;
    }

    std::vector<float> values(artifact.dimension, 0.0F);
    const auto features =
        collect_features(text, artifact.min_char_ngram,
                         artifact.max_char_ngram, artifact.include_bigrams);
    for (const auto& feature : features) {
        const std::uint64_t base = fnv1a(feature) ^ artifact.seed;
        for (std::uint8_t projection = 0;
             projection < artifact.projections_per_feature; ++projection) {
            const std::uint64_t mixed =
                mix64(base + (static_cast<std::uint64_t>(projection) *
                              0x9E3779B97F4A7C15ULL));
            const std::size_t index =
                static_cast<std::size_t>(mixed %
                                         static_cast<std::uint64_t>(artifact.dimension));
            const float sign = ((mixed >> 63U) & 1U) != 0U ? 1.0F : -1.0F;
            values[index] += sign / static_cast<float>(projection + 1U);
        }
    }

    return Vector{std::move(values)};
}

std::string_view LocalTextEmbedder::provider_name() const noexcept {
    return provider_;
}

std::string_view LocalTextEmbedder::model_name() const noexcept {
    return options_.model;
}

std::string_view LocalTextEmbedder::revision_name() const noexcept {
    return options_.revision;
}

std::string_view LocalTextEmbedder::backend_name() const noexcept {
    return backend_;
}

TextEmbedderKind LocalTextEmbedder::kind() const noexcept {
    return TextEmbedderKind::local_builtin;
}

std::uint16_t LocalTextEmbedder::output_dimension() const noexcept {
    return options_.dimension;
}

bool LocalTextEmbedder::rehydratable() const noexcept { return true; }

bool LocalTextEmbedder::loaded() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return artifact_ != nullptr;
}

void LocalTextEmbedder::unload() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    artifact_.reset();
}

std::string LocalTextEmbedder::fingerprint() const { return fingerprint_; }

std::string LocalTextEmbedder::storage_path() const { return options_.storage_path; }

TextEmbedderPtr make_local_text_embedder(const LocalTextEmbedderOptions& options) {
    return std::make_shared<LocalTextEmbedder>(options);
}

TextEmbedderPtr make_default_local_text_embedder(const std::uint16_t dimension,
                                                 const std::string& storage_path) {
    return make_local_text_embedder(LocalTextEmbedderOptions{
        .model = "default",
        .revision = "v1",
        .storage_path = storage_path,
        .dimension = dimension,
    });
}

}  // namespace elips
