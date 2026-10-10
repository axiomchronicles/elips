// Minimal consumer: exercises the installed public headers + exported library.
// Builds a flat index, adds two vectors, searches, and checks the nearest hit.
#include <cassert>
#include <cstdio>
#include <vector>

#include "elips/index_engine/IndexFactory.hpp"
#include "elips/domain/RecordID.hpp"

int main() {
    auto index = elips::index_factory(2, elips::Metric::euclidean, "Flat");
    const auto a = elips::RecordID::generate();
    const auto b = elips::RecordID::generate();
    const std::vector<float> va{0.0F, 0.0F};
    const std::vector<float> vb{10.0F, 10.0F};
    index->add_with_ids(a, va);
    index->add_with_ids(b, vb);

    const std::vector<float> query{0.1F, 0.1F};
    const auto hits = index->search(query, 1);
    assert(hits.size() == 1);
    assert(hits.front().first == a);  // nearest to (0,0)
    assert(index->ntotal() == 2);

    std::puts("elips consumer smoke: OK");
    return 0;
}
