#ifndef ELIPS_METADATA_FILTER_HPP
#define ELIPS_METADATA_FILTER_HPP

#include <memory>
#include <string>
#include <vector>

#include "elips/domain/Record.hpp"

namespace elips {

enum class Comparator { eq, ne, lt, le, gt, ge };

// A metadata predicate over a record's Payload. Supports comparisons,
// set membership, substring containment, and boolean composition. A
// default-constructed Filter matches everything.
//
// Two construction styles:
//   - fluent builder (SDK):  Filter().field("year").gte(2023).field("t").equals("x")
//     (chained leaves are AND-ed together)
//   - leaf factories + combinators (used by the EQL executor)
class Filter {
public:
    Filter() = default;

    // --- fluent builder (chained predicates are AND-ed) ---
    Filter& field(std::string name);
    Filter& equals(MetaValue value);
    Filter& not_equals(MetaValue value);
    Filter& lt(MetaValue value);
    Filter& le(MetaValue value);
    Filter& gt(MetaValue value);
    Filter& ge(MetaValue value);
    Filter& one_of(std::vector<MetaValue> values);
    Filter& contains(std::string substring);

    // --- leaf factories ---
    [[nodiscard]] static Filter compare(std::string field, Comparator op,
                                        MetaValue value);
    [[nodiscard]] static Filter in_set(std::string field,
                                       std::vector<MetaValue> values);
    [[nodiscard]] static Filter has_substring(std::string field,
                                              std::string substring);

    // --- combinators ---
    [[nodiscard]] Filter and_(const Filter& other) const;
    [[nodiscard]] Filter or_(const Filter& other) const;
    [[nodiscard]] static Filter not_(const Filter& inner);

    [[nodiscard]] bool matches(const Payload& payload) const;
    [[nodiscard]] bool matches_all() const noexcept { return root_ == nullptr; }

private:
    enum class Kind { cmp, in, contains, conj, disj, neg, none };
    struct Node {
        Kind kind;
        std::string field;
        Comparator cmp{Comparator::eq};
        MetaValue value;
        std::vector<MetaValue> set;
        std::shared_ptr<const Node> a;
        std::shared_ptr<const Node> b;
    };

    void and_leaf(std::shared_ptr<const Node> leaf);
    Filter& add_cmp(Comparator op, MetaValue value);
    static bool eval_node(const Node& node, const Payload& payload);
    static std::shared_ptr<Node> make_node(Node node);

    std::shared_ptr<const Node> root_;  // nullptr => match all
    std::string current_field_;
};

}  // namespace elips

#endif  // ELIPS_METADATA_FILTER_HPP
