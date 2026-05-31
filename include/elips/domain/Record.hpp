#ifndef ELIPS_DOMAIN_RECORD_HPP
#define ELIPS_DOMAIN_RECORD_HPP

#include <cstdint>
#include <map>
#include <string>
#include <variant>

#include "elips/domain/RecordID.hpp"
#include "elips/domain/Vector.hpp"

namespace elips {

// A single metadata value. Dynamic schema (no upfront declaration).
using MetaValue = std::variant<std::int64_t, double, bool, std::string>;

// Metadata payload attached to a record: key -> typed value.
using Payload = std::map<std::string, MetaValue>;

// A vector with identity and payload.
struct Record {
    RecordID id;
    Vector vector;
    Payload payload;
};

}  // namespace elips

#endif  // ELIPS_DOMAIN_RECORD_HPP
