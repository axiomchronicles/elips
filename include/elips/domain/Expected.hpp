#ifndef ELIPS_DOMAIN_EXPECTED_HPP
#define ELIPS_DOMAIN_EXPECTED_HPP

#include <version>
#if __has_include(<expected>)
#include <expected>
#endif

#ifndef __cpp_lib_expected
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace std {

template <typename E>
class unexpected {
public:
    constexpr unexpected(const E& e) : val_(e) {}
    constexpr unexpected(E&& e) : val_(std::move(e)) {}
    constexpr const E& error() const & noexcept { return val_; }
    constexpr E& error() & noexcept { return val_; }
    constexpr const E&& error() const && noexcept { return std::move(val_); }
    constexpr E&& error() && noexcept { return std::move(val_); }

private:
    E val_;
};

template <typename E>
unexpected(E) -> unexpected<E>;

template <typename T, typename E>
class expected {
public:
    constexpr expected() : var_(T{}) {}
    constexpr expected(const T& val) : var_(val) {}
    constexpr expected(T&& val) : var_(std::move(val)) {}
    template <typename Err>
    constexpr expected(const unexpected<Err>& u) : var_(u.error()) {}
    template <typename Err>
    constexpr expected(unexpected<Err>&& u) : var_(std::move(u.error())) {}

    constexpr bool has_value() const noexcept { return std::holds_alternative<T>(var_); }
    constexpr explicit operator bool() const noexcept { return has_value(); }

    constexpr const T& value() const & {
        if (!has_value()) throw std::runtime_error("bad expected access");
        return std::get<T>(var_);
    }
    constexpr T& value() & {
        if (!has_value()) throw std::runtime_error("bad expected access");
        return std::get<T>(var_);
    }
    constexpr const T& operator*() const & noexcept { return std::get<T>(var_); }
    constexpr T& operator*() & noexcept { return std::get<T>(var_); }
    constexpr const T* operator->() const noexcept { return &std::get<T>(var_); }
    constexpr T* operator->() noexcept { return &std::get<T>(var_); }

    constexpr const E& error() const & noexcept { return std::get<E>(var_); }
    constexpr E& error() & noexcept { return std::get<E>(var_); }

private:
    std::variant<T, E> var_;
};

template <typename E>
class expected<void, E> {
public:
    constexpr expected() : has_val_(true) {}
    template <typename Err>
    constexpr expected(const unexpected<Err>& u) : has_val_(false), err_(u.error()) {}
    template <typename Err>
    constexpr expected(unexpected<Err>&& u) : has_val_(false), err_(std::move(u.error())) {}

    constexpr bool has_value() const noexcept { return has_val_; }
    constexpr explicit operator bool() const noexcept { return has_value(); }
    constexpr void value() const {
        if (!has_value()) throw std::runtime_error(err_);
    }
    constexpr const E& error() const & noexcept { return err_; }
    constexpr E& error() & noexcept { return err_; }

private:
    bool has_val_{true};
    E err_{};
};

}  // namespace std
#endif

#endif  // ELIPS_DOMAIN_EXPECTED_HPP
