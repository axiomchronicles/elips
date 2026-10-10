// Self-check for the ELIPS C API: id round-trip + create/add/search/free.
// Exits non-zero on failure. Not a ctest (keeps the main suite count stable).
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstddef>

#include "elips/c_api/elips.h"

int main() {
    elips_index* idx = elips_index_create(2, ELIPS_METRIC_EUCLIDEAN, "Flat");
    assert(idx != nullptr);

    const float a[2] = {0.0F, 0.0F};
    const float b[2] = {10.0F, 10.0F};
    assert(elips_index_add(idx, 42, a, 2) == ELIPS_OK);
    assert(elips_index_add(idx, -7, b, 2) == ELIPS_OK);
    assert(elips_index_ntotal(idx) == 2);

    // Wrong dimension is rejected.
    assert(elips_index_add(idx, 1, a, 3) == ELIPS_ERR_INVALID_ARG);

    const float query[2] = {0.1F, 0.1F};
    int64_t ids[2] = {0, 0};
    float dists[2] = {0.0F, 0.0F};
    size_t count = 0;
    assert(elips_index_search(idx, query, 2, 2, ids, dists, &count) == ELIPS_OK);
    assert(count == 2);
    assert(ids[0] == 42);   // id round-trips through RecordID encoding
    assert(ids[1] == -7);   // negative ids survive too
    assert(dists[0] <= dists[1]);

    elips_index_free(idx);
    elips_index_free(nullptr);  // NULL-safe

    std::puts("elips C API self-check: OK");
    return 0;
}
