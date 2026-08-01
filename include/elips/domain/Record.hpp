#ifndef ELIPS_DOMAIN_RECORD_HPP
#define ELIPS_DOMAIN_RECORD_HPP

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>

#include "elips/domain/RecordID.hpp"
#include "elips/domain/Vector.hpp"
#include "elips/quant_engine/QuantParams.hpp"

namespace elips {

// A single metadata value. Dynamic schema (no upfront declaration).
using MetaValue = std::variant<std::int64_t, double, bool, std::string>;

// Metadata payload attached to a record: key -> typed value.
using Payload = std::map<std::string, MetaValue>;

struct DocumentAttachment {
    std::string text;
    std::string uri;
    std::string mime_type{"text/plain"};
};

struct ChunkInfo {
    std::string document_key;
    std::uint32_t ordinal{0};
    std::uint32_t char_start{0};
    std::uint32_t char_end{0};
};

struct EmbeddingLineage {
    std::string provider;
    std::string model;
    std::string revision;
    Payload attributes;
};

// A vector with identity and payload.
//
// In a quantized vault the compressed code is what the record actually owns:
// `codes` holds it, `codec` says which quantizer produced it, and `vector`
// carries a reconstruction for callers that asked for one. Keeping the code
// inside the record rather than in a side table is what lets checkpointing and
// transaction undo move records around without decoding -- both copy Records
// verbatim, so neither can accumulate a generation of loss.
struct Record {
    RecordID id;
    Vector vector;
    Payload payload;
    std::optional<DocumentAttachment> document;
    std::optional<ChunkInfo> chunk;
    std::optional<EmbeddingLineage> lineage;
    std::vector<std::uint8_t> codes;
    quant::CodecId codec{quant::CodecId::none};

    // True when `vector` is a reconstruction rather than the bytes that were
    // written. Surfaced to callers through SearchResult and the Python
    // StoredRecord so an approximate value is never mistaken for an exact one.
    [[nodiscard]] bool approximate() const noexcept {
        return codec != quant::CodecId::none;
    }
};

}  // namespace elips

#endif  // ELIPS_DOMAIN_RECORD_HPP
