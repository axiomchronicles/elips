// ELIPS v2 getting started (C++).
// Build with the project and run the compiled example binary.

#include <iostream>
#include <optional>

#include "elips/elips.hpp"

int main() {
    auto db = elips::open(
        ":memory:",
        elips::Config{}
            .dimension(128)
            .metric(elips::Metric::cosine));

    auto& docs = db->vault("documents");
    if (const auto info = db->config().text_embedder_info(); info.has_value()) {
        std::cout << "text embedder: " << info->provider << "/"
                  << info->model << "@" << info->revision << '\n';
    }

    elips::ChunkInfo chunk;
    chunk.document_key = "doc-alpha";
    chunk.ordinal = 0;
    chunk.char_start = 0;
    chunk.char_end = 17;

    docs.place_document("alpha design note",
                        {{"kind", std::string{"design"}}},
                        std::nullopt,
                        chunk);
    docs.place_document("beta incident runbook",
                        {{"kind", std::string{"ops"}}});

    std::cout << "text probe:\n";
    for (const auto& hit : docs.seek_text("alpha", 2)) {
        const auto title = hit.document.has_value() ? hit.document->text : "";
        std::cout << "  " << title << " distance=" << hit.distance << '\n';
    }

    std::cout << "\nhybrid probe:\n";
    for (const auto& hit :
         docs.seek_hybrid(elips::Vector{{0.0F, 1.0F}}, "alpha", 2)) {
        const auto kind = std::get<std::string>(hit.data.at("kind"));
        std::cout << "  " << kind << " distance=" << hit.distance << '\n';
    }

    const auto filter =
        elips::Filter().field("kind").equals(std::string{"design"});
    const auto plan = docs.explain_seek(elips::Vector{{1.0F, 0.0F}},
                                        1,
                                        filter,
                                        std::nullopt,
                                        true);
    std::cout << "\nplanner: candidates=" << plan.candidate_count
              << " metadata_accelerated=" << plan.metadata_accelerated
              << '\n';
    return 0;
}
