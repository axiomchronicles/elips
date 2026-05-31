#ifndef ELIPS_CLI_JSON_HPP
#define ELIPS_CLI_JSON_HPP

#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "elips/domain/Record.hpp"

// Minimal JSON reader/writer scoped to CLI export/import (records as one JSON
// object per line). Not a general JSON library; supports objects, arrays,
// strings, numbers, and booleans — exactly what a Record needs.
namespace elips::cli::json {

inline std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

inline std::string dump_value(const MetaValue& v) {
    std::ostringstream os;
    std::visit(
        [&os](const auto& x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::string>) {
                os << '"' << escape(x) << '"';
            } else if constexpr (std::is_same_v<T, bool>) {
                os << (x ? "true" : "false");
            } else {
                os << x;
            }
        },
        v);
    return os.str();
}

inline std::string dump_record(const Record& r) {
    std::ostringstream os;
    os << "{\"id\":\"" << r.id.to_string() << "\",\"vector\":[";
    const auto values = r.vector.values();
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) os << ',';
        os << values[i];
    }
    os << "],\"data\":{";
    bool first = true;
    for (const auto& [k, val] : r.payload) {
        if (!first) os << ',';
        first = false;
        os << '"' << escape(k) << "\":" << dump_value(val);
    }
    os << "}}";
    return os.str();
}

// --- minimal parser ---
class Reader {
public:
    explicit Reader(std::string_view text) : s_(text) {}

    // Parse one record object: {"id":..,"vector":[..],"data":{..}}
    Record record() {
        Record r;
        expect('{');
        bool first = true;
        while (true) {
            skip_ws();
            if (peek() == '}') { ++i_; break; }
            if (!first) expect(',');
            first = false;
            const std::string key = string();
            expect(':');
            if (key == "id") {
                r.id = RecordID::from_string(string());
            } else if (key == "vector") {
                r.vector = Vector{floats()};
            } else if (key == "data") {
                r.payload = object();
            } else {
                skip_value();
            }
        }
        return r;
    }

private:
    void skip_ws() {
        while (i_ < s_.size() &&
               std::isspace(static_cast<unsigned char>(s_[i_])) != 0) {
            ++i_;
        }
    }
    char peek() { skip_ws(); return i_ < s_.size() ? s_[i_] : '\0'; }
    void expect(char c) {
        skip_ws();
        if (i_ >= s_.size() || s_[i_] != c) {
            throw std::runtime_error(std::string("JSON: expected '") + c + "'");
        }
        ++i_;
    }
    std::string string() {
        expect('"');
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\' && i_ + 1 < s_.size()) ++i_;
            out.push_back(s_[i_++]);
        }
        expect('"');
        return out;
    }
    std::vector<float> floats() {
        std::vector<float> out;
        expect('[');
        skip_ws();
        if (peek() == ']') { ++i_; return out; }
        while (true) {
            out.push_back(static_cast<float>(number_raw().first));
            if (peek() == ',') { ++i_; continue; }
            break;
        }
        expect(']');
        return out;
    }
    std::pair<double, bool> number_raw() {  // (value, is_integer)
        skip_ws();
        const std::size_t start = i_;
        bool is_int = true;
        if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
        while (i_ < s_.size() &&
               (std::isdigit(static_cast<unsigned char>(s_[i_])) != 0)) {
            ++i_;
        }
        if (i_ < s_.size() && (s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E')) {
            is_int = false;
            ++i_;
            while (i_ < s_.size() &&
                   (std::isdigit(static_cast<unsigned char>(s_[i_])) != 0 ||
                    s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E' ||
                    s_[i_] == '-' || s_[i_] == '+')) {
                ++i_;
            }
        }
        return {std::stod(std::string(s_.substr(start, i_ - start))), is_int};
    }
    MetaValue value() {
        skip_ws();
        const char c = peek();
        if (c == '"') return MetaValue{string()};
        if (c == 't' || c == 'f') {
            const bool b = c == 't';
            i_ += b ? 4 : 5;  // skip true/false
            return MetaValue{b};
        }
        const auto [num, is_int] = number_raw();
        return is_int ? MetaValue{static_cast<std::int64_t>(num)}
                      : MetaValue{num};
    }
    Payload object() {
        Payload p;
        expect('{');
        skip_ws();
        if (peek() == '}') { ++i_; return p; }
        while (true) {
            const std::string key = string();
            expect(':');
            p.emplace(key, value());
            if (peek() == ',') { ++i_; continue; }
            break;
        }
        expect('}');
        return p;
    }
    void skip_value() { value(); }

    std::string_view s_;
    std::size_t i_{0};
};

inline Record parse_record(std::string_view line) {
    return Reader(line).record();
}

}  // namespace elips::cli::json

#endif  // ELIPS_CLI_JSON_HPP
