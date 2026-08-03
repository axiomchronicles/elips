#include "elips/metadata/MetadataIndex.hpp"

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <type_traits>

namespace elips {
namespace {

std::set<RecordID> intersect_sets(const std::set<RecordID>& left,
                                  const std::set<RecordID>& right) {
    std::set<RecordID> out;
    std::set_intersection(left.begin(), left.end(), right.begin(), right.end(),
                          std::inserter(out, out.begin()));
    return out;
}

}  // namespace

std::string MetadataIndex::encode_value(const MetaValue& value) {
    std::ostringstream encoded;
    std::visit(
        [&encoded](const auto& v) {
            using V = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<V, std::int64_t>) {
                encoded << "n:" << static_cast<double>(v);
            } else if constexpr (std::is_same_v<V, double>) {
                encoded << std::setprecision(17) << "n:" << v;
            } else if constexpr (std::is_same_v<V, bool>) {
                encoded << "b:" << (v ? '1' : '0');
            } else {
                encoded << "s:" << v;
            }
        },
        value);
    return encoded.str();
}

void MetadataIndex::insert(const RecordID& id, const Payload& payload) {
    for (const auto& [field, value] : payload) {
        exact_[field][encode_value(value)].insert(id);
    }
}

void MetadataIndex::remove(const RecordID& id, const Payload& payload) noexcept {
    for (const auto& [field, value] : payload) {
        const auto field_it = exact_.find(field);
        if (field_it == exact_.end()) {
            continue;
        }
        const auto value_it = field_it->second.find(encode_value(value));
        if (value_it == field_it->second.end()) {
            continue;
        }
        value_it->second.erase(id);
        if (value_it->second.empty()) {
            field_it->second.erase(value_it);
        }
        if (field_it->second.empty()) {
            exact_.erase(field_it);
        }
    }
}

std::optional<std::set<RecordID>> MetadataIndex::exact_candidates(
    const Filter& filter) const {
    const auto constraints = filter.exact_constraints();
    if (!constraints.has_value()) {
        return std::nullopt;
    }
    if (constraints->empty()) {
        return std::set<RecordID>{};
    }

    std::optional<std::set<RecordID>> current;
    for (const auto& [field, values] : *constraints) {
        std::set<RecordID> field_matches;
        const auto field_it = exact_.find(field);
        if (field_it != exact_.end()) {
            for (const auto& value : values) {
                const auto value_it = field_it->second.find(encode_value(value));
                if (value_it != field_it->second.end()) {
                    field_matches.insert(value_it->second.begin(),
                                         value_it->second.end());
                }
            }
        }
        if (!current.has_value()) {
            current = std::move(field_matches);
        } else {
            *current = intersect_sets(std::move(*current), field_matches);
        }
        if (current->empty()) {
            break;
        }
    }
    return current;
}

}  // namespace elips
