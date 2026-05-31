#include "elips/query_engine/QueryExecutor.hpp"

#include <algorithm>
#include <limits>

#include "elips/domain/Errors.hpp"
#include "elips/elips.hpp"
#include "elips/query_engine/EQLParser.hpp"

namespace elips::eql {
namespace {

constexpr int default_top = 10;
constexpr std::size_t unbounded_top = 100000;

Vector resolve_query(const VectorRef& ref,
                     const std::map<std::string, Vector>& bindings) {
    if (!ref.binding.empty()) {
        const auto it = bindings.find(ref.binding);
        if (it == bindings.end()) {
            throw NotFound{"EQL: unbound query vector $" + ref.binding};
        }
        return it->second;
    }
    return Vector{ref.literal};
}

bool meta_less(const MetaValue& a, const MetaValue& b) {
    auto as_num = [](const MetaValue& v, double& out) {
        if (std::holds_alternative<std::int64_t>(v)) {
            out = static_cast<double>(std::get<std::int64_t>(v));
            return true;
        }
        if (std::holds_alternative<double>(v)) {
            out = std::get<double>(v);
            return true;
        }
        return false;
    };
    double x = 0;
    double y = 0;
    if (as_num(a, x) && as_num(b, y)) {
        return x < y;
    }
    if (std::holds_alternative<std::string>(a) &&
        std::holds_alternative<std::string>(b)) {
        return std::get<std::string>(a) < std::get<std::string>(b);
    }
    return a.index() < b.index();
}

void apply_projection(std::vector<SearchResult>& results,
                      const std::vector<std::string>& fields) {
    if (fields.empty()) {
        return;
    }
    for (auto& result : results) {
        Payload kept;
        for (const auto& field : fields) {
            const auto it = result.data.find(field);
            if (it != result.data.end()) {
                kept.emplace(field, it->second);
            }
        }
        result.data = std::move(kept);
    }
}

struct Executor {
    ElipsInstance& db;
    const std::map<std::string, Vector>& bindings;

    std::vector<SearchResult> operator()(const SearchStatement& s) const {
        const Vector query = resolve_query(s.query, bindings);
        const std::optional<float> threshold =
            s.threshold ? std::optional<float>{static_cast<float>(*s.threshold)}
                        : std::nullopt;
        const std::size_t top =
            s.top ? static_cast<std::size_t>(*s.top)
                  : (s.threshold ? unbounded_top
                                 : static_cast<std::size_t>(default_top));
        auto results = db.vault(s.vault).seek(query, top, s.where, threshold);

        if (s.rank_by) {
            const std::string& field = *s.rank_by;
            std::stable_sort(results.begin(), results.end(),
                             [&](const SearchResult& a, const SearchResult& b) {
                                 const auto ia = a.data.find(field);
                                 const auto ib = b.data.find(field);
                                 if (ia == a.data.end()) return false;
                                 if (ib == b.data.end()) return true;
                                 return meta_less(ia->second, ib->second);
                             });
        }
        apply_projection(results, s.projection);
        return results;
    }

    std::vector<SearchResult> operator()(const FetchStatement& s) const {
        const auto record = db.vault(s.vault).fetch(RecordID::from_string(s.id));
        if (!record) {
            return {};
        }
        return {SearchResult{record->id, 0.0F, record->payload}};
    }

    std::vector<SearchResult> operator()(const ScanStatement& s) const {
        const std::size_t offset =
            s.offset ? static_cast<std::size_t>(*s.offset) : 0;
        const std::size_t limit =
            s.limit ? static_cast<std::size_t>(*s.limit)
                    : std::numeric_limits<std::size_t>::max();
        const auto records = db.vault(s.vault).scan(s.where, offset, limit);
        std::vector<SearchResult> results;
        results.reserve(records.size());
        for (const auto& record : records) {
            results.push_back(SearchResult{record.id, 0.0F, record.payload});
        }
        return results;
    }

    std::vector<SearchResult> operator()(const InsertStatement& s) const {
        const RecordID id =
            db.vault(s.vault).place(Vector{s.vector}, s.data);
        return {SearchResult{id, 0.0F, {}}};
    }

    std::vector<SearchResult> operator()(const DeleteStatement& s) const {
        db.vault(s.vault).erase(RecordID::from_string(s.id));
        return {};
    }
};

}  // namespace

std::vector<SearchResult> execute(const Statement& statement, ElipsInstance& db,
                                  const std::map<std::string, Vector>& bindings) {
    return std::visit(Executor{db, bindings}, statement);
}

}  // namespace elips::eql

namespace elips {

std::vector<SearchResult> ElipsInstance::query(
    const std::string& eql, const std::map<std::string, Vector>& bindings) {
    return eql::execute(eql::parse(eql), *this, bindings);
}

}  // namespace elips
