// ELIPS getting started (C++). Build with the project, then:
//   c++ -std=c++23 -Iinclude examples/cpp/01_getting_started.cpp \
//       build/libelips_core.a -o gs && ./gs
#include <iostream>

#include "elips/elips.hpp"

int main() {
    auto db = elips::open(
        ":memory:", elips::Config{}.dimension(3).metric(elips::Metric::cosine));
    auto& docs = db->vault("documents");

    docs.place(elips::Vector{{1.0F, 0.0F, 0.0F}},
               {{"title", std::string{"alpha"}}, {"year", std::int64_t{2024}}});
    docs.place(elips::Vector{{0.0F, 1.0F, 0.0F}},
               {{"title", std::string{"beta"}}, {"year", std::int64_t{2019}}});

    std::cout << "nearest to [1, 0, 0]:\n";
    for (const auto& hit : docs.seek(elips::Vector{{0.9F, 0.1F, 0.0F}}, 2)) {
        std::cout << "  " << std::get<std::string>(hit.data.at("title")) << "  "
                  << hit.distance << '\n';
    }

    const auto recent = elips::Filter().field("year").ge(std::int64_t{2023});
    std::cout << "filtered (year >= 2023):\n";
    for (const auto& hit : docs.seek(elips::Vector{{1.0F, 0.0F, 0.0F}}, 5, recent)) {
        std::cout << "  " << std::get<std::string>(hit.data.at("title")) << '\n';
    }
    return 0;
}
