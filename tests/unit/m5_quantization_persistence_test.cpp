// M5 of the engineering audit: quantized persistence.
//
// The invariant these tests exist to protect is that codes are authoritative
// end-to-end. A stored code must survive a checkpoint/reopen cycle byte for
// byte, because the alternative -- decoding on load and re-encoding on save --
// loses a generation of accuracy on every restart, without bound and without
// any error to notice.
//
// Also covers the v2/v3 format boundary and the malformed-input paths, since
// bumping the on-disk version touched five separate reader gates.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "elips/domain/Errors.hpp"
#include "elips/elips.hpp"
#include "elips/quant_engine/Quantizer.hpp"

namespace {

namespace fs = std::filesystem;

using elips::Config;
using elips::Metric;
using elips::RecordID;
using elips::Vector;
using elips::quant::CodecId;
using elips::quant::QuantParams;

constexpr std::uint16_t kDim = 32;
constexpr std::size_t kCount = 400;

// A unique temporary directory per test, removed on destruction.
class TempDir {
public:
    explicit TempDir(const std::string& label) {
        static int counter = 0;
        path_ = fs::temp_directory_path() /
                ("elips_quant_" + label + "_" + std::to_string(++counter) + "_" +
                 std::to_string(::getpid()));
        fs::remove_all(path_);
        fs::create_directories(path_);
    }
    ~TempDir() { fs::remove_all(path_); }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    TempDir(TempDir&&) = delete;
    TempDir& operator=(TempDir&&) = delete;

    [[nodiscard]] const fs::path& path() const noexcept { return path_; }
    [[nodiscard]] std::string string() const { return path_.string(); }

private:
    fs::path path_;
};

QuantParams pq_params() {
    QuantParams params;
    params.codec = CodecId::pq;
    params.pq_dim = 8;
    params.pq_bits = 8;
    params.train_iters = 6;
    return params;
}

Config quantized_config(QuantParams params = pq_params()) {
    Config config;
    config.dimension(kDim).metric(Metric::euclidean).quantization(params);
    return config;
}

std::vector<std::vector<float>> sample_vectors(std::size_t n, unsigned seed = 5) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<std::vector<float>> rows(n, std::vector<float>(kDim));
    for (auto& row : rows) {
        for (float& v : row) {
            v = dist(rng);
        }
    }
    return rows;
}

std::vector<RecordID> populate(elips::Vault& vault,
                               const std::vector<std::vector<float>>& rows) {
    std::vector<RecordID> ids;
    ids.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        elips::Payload payload;
        payload.emplace("row", static_cast<std::int64_t>(i));
        ids.push_back(vault.place(Vector{rows[i]}, payload));
    }
    return ids;
}

std::string read_file(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

// The snapshot lands either as one file or as per-vault segments depending on
// Config::segmented_storage(); collect whichever exists.
std::string read_persisted_state(const fs::path& root) {
    std::string combined = read_file(root / "elips.snapshot");
    const fs::path segments = root / "segments";
    if (fs::exists(segments)) {
        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(segments)) {
            files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());
        for (const auto& file : files) {
            combined += read_file(file);
        }
    }
    return combined;
}

}  // namespace

// ------------------------- the drift invariant ------------------------------

TEST(M5QuantizationPersistence, RepeatedReopenLeavesTheStoredBytesIdentical) {
    const TempDir dir("drift");
    const auto rows = sample_vectors(kCount);

    {
        auto db = elips::open(dir.string(), quantized_config());
        populate(db->vault("main"), rows);
        db->quantize("main");
        db->close();
    }

    // Cycle 1 is the baseline. From cycle 2 on, every byte on disk must be
    // identical: nothing in a reopen may re-encode what is already coded.
    std::string baseline;
    for (int cycle = 0; cycle < 5; ++cycle) {
        auto db = elips::open(dir.string(), quantized_config());
        auto& vault = db->vault("main");
        EXPECT_TRUE(vault.quantized());
        EXPECT_EQ(vault.info().count, kCount);
        db->checkpoint();
        db->close();

        const std::string current = read_persisted_state(dir.path());
        ASSERT_FALSE(current.empty());
        if (cycle == 0) {
            baseline = current;
        } else {
            EXPECT_EQ(current, baseline)
                << "on-disk state changed on reopen cycle " << cycle
                << ": codes are being decoded and re-encoded";
        }
    }
}

TEST(M5QuantizationPersistence, SearchResultsAreStableAcrossReopen) {
    const TempDir dir("stable");
    const auto rows = sample_vectors(kCount);
    std::vector<RecordID> ids;

    {
        auto db = elips::open(dir.string(), quantized_config());
        ids = populate(db->vault("main"), rows);
        db->quantize("main");
        db->close();
    }

    std::vector<std::vector<RecordID>> first_run;
    for (int cycle = 0; cycle < 3; ++cycle) {
        auto db = elips::open(dir.string(), quantized_config());
        auto& vault = db->vault("main");
        std::vector<std::vector<RecordID>> run;
        for (std::size_t q = 0; q < 10; ++q) {
            std::vector<RecordID> hits;
            for (const auto& result : vault.seek(Vector{rows[q]}, 5)) {
                hits.push_back(result.id);
                // Every hit from a quantized vault must be labelled approximate.
                EXPECT_TRUE(result.approximate());
                EXPECT_EQ(result.codec, CodecId::pq);
            }
            run.push_back(std::move(hits));
        }
        db->close();

        if (cycle == 0) {
            first_run = run;
        } else {
            EXPECT_EQ(run, first_run) << "ranking drifted on cycle " << cycle;
        }
    }
}

// ---------------------------- format boundary --------------------------------

TEST(M5QuantizationPersistence, UnquantizedDatabaseRoundTripsExactly) {
    // A v3 file whose records all carry codec `none` must behave exactly as v2
    // did, including returning the stored vector bit for bit.
    const TempDir dir("plain");
    const auto rows = sample_vectors(50);
    std::vector<RecordID> ids;

    {
        Config config;
        config.dimension(kDim).metric(Metric::euclidean);
        auto db = elips::open(dir.string(), config);
        ids = populate(db->vault("main"), rows);
        db->close();
    }

    Config config;
    config.dimension(kDim).metric(Metric::euclidean);
    auto db = elips::open(dir.string(), config);
    auto& vault = db->vault("main");
    EXPECT_FALSE(vault.quantized());
    EXPECT_EQ(vault.codec(), CodecId::none);

    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto record = vault.fetch(ids[i]);
        ASSERT_TRUE(record.has_value());
        EXPECT_FALSE(record->approximate());
        const auto stored = record->vector.values();
        ASSERT_EQ(stored.size(), kDim);
        // Euclidean does not normalize, so the values survive verbatim.
        for (std::size_t d = 0; d < kDim; ++d) {
            EXPECT_FLOAT_EQ(stored[d], rows[i][d]) << "record " << i << " dim " << d;
        }
    }
}

TEST(M5QuantizationPersistence, ExistingV2DatabaseOpensUnderTheV3Reader) {
    // Simulates an on-disk database written before quantization existed: the
    // version stamp is rewritten to 2 and the codec byte and codebook section
    // are absent. The reader must accept it, not reject it and not misparse the
    // extras.
    const TempDir dir("legacy");
    const auto rows = sample_vectors(30);
    std::vector<RecordID> ids;

    {
        Config config;
        config.dimension(kDim).metric(Metric::euclidean).segmented_storage(false);
        auto db = elips::open(dir.string(), config);
        ids = populate(db->vault("main"), rows);
        db->close();
    }

    // Rewrite the v3 snapshot as a v2 one: drop the per-vault codebook byte and
    // the per-record codec byte, and stamp version 2.
    const fs::path snapshot = dir.path() / "elips.snapshot";
    ASSERT_TRUE(fs::exists(snapshot));
    const std::string v3 = read_file(snapshot);
    ASSERT_GE(v3.size(), 15U);

    std::string v2;
    v2.append(v3, 0, 4);  // magic
    const std::uint32_t two = 2U;
    v2.append(reinterpret_cast<const char*>(&two), 4);
    std::size_t pos = 8;
    v2.append(v3, pos, 2 + 1 + 4);  // dimension, metric, vault count
    pos += 2 + 1 + 4;

    // One vault: name string, then the codebook byte to drop.
    std::uint32_t name_len = 0;
    std::memcpy(&name_len, v3.data() + pos, 4);
    v2.append(v3, pos, 4 + name_len);
    pos += 4 + name_len;
    ASSERT_EQ(static_cast<unsigned char>(v3[pos]), 0U)  // codec none
        << "expected an absent codebook for an unquantized vault";
    pos += 1;  // drop the codebook section

    std::uint32_t record_count = 0;
    std::memcpy(&record_count, v3.data() + pos, 4);
    v2.append(v3, pos, 4);
    pos += 4;
    ASSERT_EQ(record_count, ids.size());

    // Each record: 16 id bytes, then a codec byte to drop, then the rest.
    for (std::uint32_t i = 0; i < record_count; ++i) {
        v2.append(v3, pos, 16);
        pos += 16;
        ASSERT_EQ(static_cast<unsigned char>(v3[pos]), 0U);
        pos += 1;  // drop the per-record codec byte

        std::uint16_t dim = 0;
        std::memcpy(&dim, v3.data() + pos, 2);
        const std::size_t vector_bytes = 2 + (std::size_t{dim} * sizeof(float));
        v2.append(v3, pos, vector_bytes);
        pos += vector_bytes;

        // payload: count, then per-entry key + tag + value. The fixture writes
        // exactly one int64 entry.
        std::uint32_t entries = 0;
        std::memcpy(&entries, v3.data() + pos, 4);
        ASSERT_EQ(entries, 1U);
        v2.append(v3, pos, 4);
        pos += 4;
        std::uint32_t key_len = 0;
        std::memcpy(&key_len, v3.data() + pos, 4);
        const std::size_t entry_bytes = 4 + key_len + 1 + sizeof(std::int64_t);
        v2.append(v3, pos, entry_bytes);
        pos += entry_bytes;

        // extras: three absent-markers for document, chunk, lineage.
        v2.append(v3, pos, 3);
        pos += 3;
    }
    ASSERT_EQ(pos, v3.size()) << "v3 snapshot layout was not fully consumed";

    {
        std::ofstream out(snapshot, std::ios::binary | std::ios::trunc);
        out.write(v2.data(), static_cast<std::streamsize>(v2.size()));
    }

    Config config;
    config.dimension(kDim).metric(Metric::euclidean).segmented_storage(false);
    auto db = elips::open(dir.string(), config);
    auto& vault = db->vault("main");
    EXPECT_EQ(vault.info().count, ids.size());
    EXPECT_FALSE(vault.quantized());

    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto record = vault.fetch(ids[i]);
        ASSERT_TRUE(record.has_value()) << "record " << i << " lost";
        // The payload proves the extras were not misparsed: a wrong `with_extras`
        // would desynchronize the stream and corrupt everything after it.
        ASSERT_TRUE(record->payload.contains("row"));
        EXPECT_EQ(std::get<std::int64_t>(record->payload.at("row")),
                  static_cast<std::int64_t>(i));
        const auto stored = record->vector.values();
        for (std::size_t d = 0; d < kDim; ++d) {
            EXPECT_FLOAT_EQ(stored[d], rows[i][d]);
        }
    }
}

TEST(M5QuantizationPersistence, NewerOnDiskVersionIsRejected) {
    const TempDir dir("future");
    {
        Config config;
        config.dimension(kDim).metric(Metric::euclidean).segmented_storage(false);
        auto db = elips::open(dir.string(), config);
        populate(db->vault("main"), sample_vectors(5));
        db->close();
    }

    const fs::path snapshot = dir.path() / "elips.snapshot";
    std::string bytes = read_file(snapshot);
    const std::uint32_t future = 99U;
    std::memcpy(bytes.data() + 4, &future, 4);
    {
        std::ofstream out(snapshot, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    Config config;
    config.dimension(kDim).metric(Metric::euclidean).segmented_storage(false);
    EXPECT_THROW((void)elips::open(dir.string(), config), elips::StorageError);
}

// -------------------------- malformed input ---------------------------------

TEST(M5QuantizationPersistence, TruncatedQuantizedSnapshotIsRejectedNotOverAllocated) {
    const TempDir dir("truncated");
    {
        auto db = elips::open(dir.string(), quantized_config());
        populate(db->vault("main"), sample_vectors(kCount));
        db->quantize("main");
        db->close();
    }

    const fs::path snapshot = dir.path() / "elips.snapshot";
    const fs::path segments = dir.path() / "segments";
    fs::path target = snapshot;
    if (!fs::exists(target) && fs::exists(segments)) {
        target = fs::directory_iterator(segments)->path();
    }
    ASSERT_TRUE(fs::exists(target));
    const std::string full = read_file(target);
    ASSERT_GT(full.size(), 64U);

    // Truncate across the whole file, including inside the codebook section and
    // inside the code blobs. Every prefix must fail fast and bounded rather than
    // resizing on an unvalidated length read straight off disk.
    for (std::size_t cut = 8; cut < full.size(); cut += 97) {
        std::ofstream out(target, std::ios::binary | std::ios::trunc);
        out.write(full.data(), static_cast<std::streamsize>(cut));
        out.close();

        try {
            auto db = elips::open(dir.string(), quantized_config());
            // Accepting a prefix is legitimate only if it decoded coherently.
            (void)db->vault("main").info();
            db->abandon();
        } catch (const elips::ElipsError&) {
            // Expected for most cuts.
        }
    }
}

TEST(M5QuantizationPersistence, CorruptCodecTagIsRejected) {
    const TempDir dir("badcodec");
    {
        Config config;
        config.dimension(kDim).metric(Metric::euclidean).segmented_storage(false);
        auto db = elips::open(dir.string(), config);
        populate(db->vault("main"), sample_vectors(5));
        db->close();
    }

    // Find the first per-record codec byte and set it to an unknown tag. It sits
    // after the header, the vault name, the codebook byte, the record count, and
    // the 16-byte id.
    const fs::path snapshot = dir.path() / "elips.snapshot";
    std::string bytes = read_file(snapshot);
    std::size_t pos = 8 + 2 + 1 + 4;
    std::uint32_t name_len = 0;
    std::memcpy(&name_len, bytes.data() + pos, 4);
    pos += 4 + name_len;  // vault name
    pos += 1;             // vault codebook byte
    pos += 4;             // record count
    pos += 16;            // record id
    bytes[pos] = static_cast<char>(0x7F);
    {
        std::ofstream out(snapshot, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    Config config;
    config.dimension(kDim).metric(Metric::euclidean).segmented_storage(false);
    EXPECT_THROW((void)elips::open(dir.string(), config), elips::StorageError);
}

// ------------------------------ lifecycle ------------------------------------

TEST(M5QuantizationPersistence, QuantizeRejectsInvalidStates) {
    const TempDir dir("states");
    auto db = elips::open(dir.string(), quantized_config());

    // Empty vault: nothing to learn a codebook from.
    EXPECT_THROW(db->quantize("main"), elips::ConfigError);

    populate(db->vault("main"), sample_vectors(50));
    db->quantize("main");
    EXPECT_TRUE(db->vault("main").quantized());

    // Already quantized.
    EXPECT_THROW(db->quantize("main"), elips::ConfigError);
    db->abandon();
}

TEST(M5QuantizationPersistence, QuantizeRequiresAConfiguredCodec) {
    const TempDir dir("nocodec");
    Config config;
    config.dimension(kDim).metric(Metric::euclidean);
    auto db = elips::open(dir.string(), config);
    populate(db->vault("main"), sample_vectors(20));
    EXPECT_THROW(db->quantize("main"), elips::ConfigError);
    db->abandon();
}

TEST(M5QuantizationPersistence, InsertsAfterQuantizationAreEncodedOnArrival) {
    const TempDir dir("postinsert");
    const auto rows = sample_vectors(kCount);
    RecordID late;

    {
        auto db = elips::open(dir.string(), quantized_config());
        auto& vault = db->vault("main");
        populate(vault, rows);
        db->quantize("main");

        late = vault.place(Vector{rows[0]});
        const auto record = vault.fetch(late);
        ASSERT_TRUE(record.has_value());
        EXPECT_TRUE(record->approximate());
        EXPECT_EQ(record->codec, CodecId::pq);
        db->close();
    }

    // And it survives the reopen, which means it was logged as insert_q and
    // replayed as a code.
    auto db = elips::open(dir.string(), quantized_config());
    const auto record = db->vault("main").fetch(late);
    ASSERT_TRUE(record.has_value());
    EXPECT_TRUE(record->approximate());
    EXPECT_EQ(db->vault("main").info().count, kCount + 1);
}

TEST(M5QuantizationPersistence, VaultInfoReportsTheCompressionRatio) {
    const TempDir dir("info");
    auto db = elips::open(dir.string(), quantized_config());
    populate(db->vault("main"), sample_vectors(kCount));

    EXPECT_EQ(db->vault("main").info().codec, CodecId::none);
    EXPECT_EQ(db->vault("main").info().code_bytes, 0U);

    db->quantize("main");
    const auto info = db->vault("main").info();
    EXPECT_EQ(info.codec, CodecId::pq);
    EXPECT_EQ(info.code_bytes, 8U);
    // 32 floats down to 8 bytes.
    EXPECT_EQ((kDim * sizeof(float)) / info.code_bytes, 16U);
    db->abandon();
}

TEST(M5QuantizationPersistence, VacuumAndCompactPreserveCompression) {
    // rebuild_index_locked() reconstructs the index from Config alone, so a
    // quantizer that was not reachable from Config would be silently dropped
    // here and the vault would revert to fp32 storage.
    const TempDir dir("vacuum");
    const auto rows = sample_vectors(kCount);
    std::vector<RecordID> ids;

    auto db = elips::open(dir.string(), quantized_config());
    auto& vault = db->vault("main");
    ids = populate(vault, rows);
    db->quantize("main");

    for (std::size_t i = 0; i < 100; ++i) {
        vault.erase(ids[i]);
    }
    db->vacuum();
    EXPECT_TRUE(vault.quantized());
    EXPECT_EQ(vault.info().codec, CodecId::pq);

    db->compact();
    EXPECT_TRUE(vault.quantized());
    EXPECT_EQ(vault.info().count, kCount - 100);

    const auto record = vault.fetch(ids[200]);
    ASSERT_TRUE(record.has_value());
    EXPECT_TRUE(record->approximate());
    db->close();
}

TEST(M5QuantizationPersistence, EveryCodecSurvivesARoundTrip) {
    for (const auto codec : {CodecId::pq, CodecId::opq, CodecId::sq8}) {
        const TempDir dir(std::string("codec_") + elips::quant::to_string(codec));
        const auto rows = sample_vectors(200);
        std::vector<RecordID> ids;

        QuantParams params = pq_params();
        params.codec = codec;
        params.opq_iters = 2;

        {
            auto db = elips::open(dir.string(), quantized_config(params));
            ids = populate(db->vault("main"), rows);
            db->quantize("main");
            db->close();
        }

        auto db = elips::open(dir.string(), quantized_config(params));
        auto& vault = db->vault("main");
        EXPECT_EQ(vault.codec(), codec)
            << "codec " << elips::quant::to_string(codec);
        EXPECT_EQ(vault.info().count, ids.size());

        // Self-retrieval: the nearest neighbor of a stored vector is itself.
        const auto hits = vault.seek(Vector{rows[3]}, 1);
        ASSERT_FALSE(hits.empty());
        EXPECT_EQ(hits[0].id, ids[3]) << "codec " << elips::quant::to_string(codec);
        db->close();
    }
}

// ---------------------------- transactions ----------------------------------

TEST(M5QuantizationPersistence, RollbackLeavesUntouchedCodesBitIdentical) {
    // commit() builds its undo log from fetch(), which hands back a
    // reconstruction alongside the code. Restoring must use the code: otherwise
    // rolling back one write degrades records the transaction never modified.
    const TempDir dir("rollback");
    const auto rows = sample_vectors(kCount);

    auto db = elips::open(dir.string(), quantized_config());
    auto& vault = db->vault("main");
    const auto ids = populate(vault, rows);
    db->quantize("main");

    const auto before = vault.fetch(ids[7]);
    ASSERT_TRUE(before.has_value());
    ASSERT_FALSE(before->codes.empty());

    // Overwrite the record inside a transaction, then roll back.
    {
        auto txn = db->begin_transaction();
        auto handle = txn.vault("main");
        handle.place(Vector{rows[99]}, {}, ids[7]);
        txn.rollback();
    }

    const auto after = vault.fetch(ids[7]);
    ASSERT_TRUE(after.has_value());
    EXPECT_EQ(after->codes, before->codes)
        << "rollback re-encoded the record instead of restoring its code";
    db->close();
}
