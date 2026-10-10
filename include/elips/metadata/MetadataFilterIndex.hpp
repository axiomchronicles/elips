#ifndef ELIPS_METADATA_METADATA_FILTER_INDEX_HPP
#define ELIPS_METADATA_METADATA_FILTER_INDEX_HPP

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "elips/domain/Record.hpp"
#include "elips/domain/RecordID.hpp"
#include "elips/metadata/Filter.hpp"

namespace elips {

class MetadataFilterIndex {
public:
    void insert(const RecordID& id, const Payload& payload);
    void remove(const RecordID& id, const Payload& payload) noexcept;
    [[nodiscard]] std::optional<std::set<RecordID>>
    exact_candidates(const Filter& filter) const;

private:
    [[nodiscard]] static std::string encode_value(const MetaValue& value);

    std::map<std::string, std::map<std::string, std::set<RecordID>>> exact_;
};

using MetadataIndex [[deprecated("renamed to MetadataFilterIndex")]] = MetadataFilterIndex;

}  // namespace elips

#endif  // ELIPS_METADATA_METADATA_FILTER_INDEX_HPP
