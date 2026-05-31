#ifndef ELIPS_DOMAIN_SEARCH_RESULT_HPP
#define ELIPS_DOMAIN_SEARCH_RESULT_HPP

#include "elips/domain/Record.hpp"
#include "elips/domain/RecordID.hpp"

namespace elips {

// One hit from a seek(): the record identity, its distance to the query
// (smaller = closer, ordering-normalized across metrics), and its payload.
struct SearchResult {
    RecordID id;
    float distance{0.0F};
    Payload data;
};

}  // namespace elips

#endif  // ELIPS_DOMAIN_SEARCH_RESULT_HPP
