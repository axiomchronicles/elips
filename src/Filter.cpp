#include "elips/metadata/Filter.hpp"

#include <optional>

namespace elips {
namespace {

// Three-way compare of two metadata values. Numeric types compare across
// int64/double; bool and string compare within type. Returns nullopt when the
// values are not comparable (different kinds).
std::optional<int> compare_values(const MetaValue& a, const MetaValue& b) {
    const bool a_num = std::holds_alternative<std::int64_t>(a) ||
                       std::holds_alternative<double>(a);
    const bool b_num = std::holds_alternative<std::int64_t>(b) ||
                       std::holds_alternative<double>(b);
    if (a_num && b_num) {
        const double x = std::holds_alternative<std::int64_t>(a)
                             ? static_cast<double>(std::get<std::int64_t>(a))
                             : std::get<double>(a);
        const double y = std::holds_alternative<std::int64_t>(b)
                             ? static_cast<double>(std::get<std::int64_t>(b))
                             : std::get<double>(b);
        if (x < y) return -1;
        if (x > y) return 1;
        return 0;
    }
    if (std::holds_alternative<bool>(a) && std::holds_alternative<bool>(b)) {
        return static_cast<int>(std::get<bool>(a)) -
               static_cast<int>(std::get<bool>(b));
    }
    if (std::holds_alternative<std::string>(a) &&
        std::holds_alternative<std::string>(b)) {
        return std::get<std::string>(a).compare(std::get<std::string>(b));
    }
    return std::nullopt;
}

bool apply_comparator(Comparator op, int ordering) {
    switch (op) {
        case Comparator::eq: return ordering == 0;
        case Comparator::ne: return ordering != 0;
        case Comparator::lt: return ordering < 0;
        case Comparator::le: return ordering <= 0;
        case Comparator::gt: return ordering > 0;
        case Comparator::ge: return ordering >= 0;
    }
    return false;
}

}  // namespace

std::shared_ptr<Filter::Node> Filter::make_node(Node node) {
    return std::make_shared<Node>(std::move(node));
}

bool Filter::eval_node(const Node& node, const Payload& payload) {
    switch (node.kind) {
        case Kind::cmp: {
            const auto it = payload.find(node.field);
            if (it == payload.end()) {
                return false;
            }
            const auto ord = compare_values(it->second, node.value);
            return ord.has_value() && apply_comparator(node.cmp, *ord);
        }
        case Kind::in: {
            const auto it = payload.find(node.field);
            if (it == payload.end()) {
                return false;
            }
            for (const auto& candidate : node.set) {
                const auto ord = compare_values(it->second, candidate);
                if (ord.has_value() && *ord == 0) {
                    return true;
                }
            }
            return false;
        }
        case Kind::contains: {
            const auto it = payload.find(node.field);
            if (it == payload.end() ||
                !std::holds_alternative<std::string>(it->second)) {
                return false;
            }
            return std::get<std::string>(it->second)
                       .find(std::get<std::string>(node.value)) !=
                   std::string::npos;
        }
        case Kind::conj:
            return eval_node(*node.a, payload) && eval_node(*node.b, payload);
        case Kind::disj:
            return eval_node(*node.a, payload) || eval_node(*node.b, payload);
        case Kind::neg:
            return !eval_node(*node.a, payload);
        case Kind::none:
            return false;
    }
    return false;
}

void Filter::and_leaf(std::shared_ptr<const Node> leaf) {
    if (root_ == nullptr) {
        root_ = std::move(leaf);
        return;
    }
    root_ = make_node(Node{Kind::conj, {}, {}, {}, {}, root_, std::move(leaf)});
}

Filter& Filter::add_cmp(Comparator op, MetaValue value) {
    and_leaf(make_node(Node{Kind::cmp, current_field_, op, std::move(value), {},
                            nullptr, nullptr}));
    return *this;
}

Filter& Filter::field(std::string name) {
    current_field_ = std::move(name);
    return *this;
}

Filter& Filter::equals(MetaValue value) { return add_cmp(Comparator::eq, std::move(value)); }
Filter& Filter::not_equals(MetaValue value) { return add_cmp(Comparator::ne, std::move(value)); }
Filter& Filter::lt(MetaValue value) { return add_cmp(Comparator::lt, std::move(value)); }
Filter& Filter::le(MetaValue value) { return add_cmp(Comparator::le, std::move(value)); }
Filter& Filter::gt(MetaValue value) { return add_cmp(Comparator::gt, std::move(value)); }
Filter& Filter::ge(MetaValue value) { return add_cmp(Comparator::ge, std::move(value)); }

Filter& Filter::one_of(std::vector<MetaValue> values) {
    and_leaf(make_node(Node{Kind::in, current_field_, Comparator::eq, {},
                            std::move(values), nullptr, nullptr}));
    return *this;
}

Filter& Filter::contains(std::string substring) {
    and_leaf(make_node(Node{Kind::contains, current_field_, Comparator::eq,
                            std::move(substring), {}, nullptr, nullptr}));
    return *this;
}

Filter Filter::compare(std::string field, Comparator op, MetaValue value) {
    Filter f;
    f.and_leaf(make_node(Node{Kind::cmp, std::move(field), op, std::move(value),
                              {}, nullptr, nullptr}));
    return f;
}

Filter Filter::in_set(std::string field, std::vector<MetaValue> values) {
    Filter f;
    f.and_leaf(make_node(Node{Kind::in, std::move(field), Comparator::eq, {},
                              std::move(values), nullptr, nullptr}));
    return f;
}

Filter Filter::has_substring(std::string field, std::string substring) {
    Filter f;
    f.and_leaf(make_node(Node{Kind::contains, std::move(field), Comparator::eq,
                              std::move(substring), {}, nullptr, nullptr}));
    return f;
}

Filter Filter::and_(const Filter& other) const {
    if (root_ == nullptr) return other;
    if (other.root_ == nullptr) return *this;
    Filter f;
    f.root_ = make_node(Node{Kind::conj, {}, {}, {}, {}, root_, other.root_});
    return f;
}

Filter Filter::or_(const Filter& other) const {
    if (root_ == nullptr || other.root_ == nullptr) {
        return Filter{};  // disjunction with match-all is match-all
    }
    Filter f;
    f.root_ = make_node(Node{Kind::disj, {}, {}, {}, {}, root_, other.root_});
    return f;
}

Filter Filter::not_(const Filter& inner) {
    Filter f;
    if (inner.root_ == nullptr) {
        // NOT(match-all) matches nothing.
        f.root_ = make_node(Node{Kind::none, {}, {}, {}, {}, nullptr, nullptr});
        return f;
    }
    f.root_ = make_node(Node{Kind::neg, {}, {}, {}, {}, inner.root_, nullptr});
    return f;
}

bool Filter::matches(const Payload& payload) const {
    return root_ == nullptr || eval_node(*root_, payload);
}

}  // namespace elips
