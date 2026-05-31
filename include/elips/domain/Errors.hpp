#ifndef ELIPS_DOMAIN_ERRORS_HPP
#define ELIPS_DOMAIN_ERRORS_HPP

#include <stdexcept>

namespace elips {

// Base for all ELIPS errors (E.14: purpose-designed exception types).
class ElipsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Vector dimension does not match the database/vault configuration.
class DimensionMismatch : public ElipsError {
public:
    using ElipsError::ElipsError;
};

// Vector contains NaN/Inf or is otherwise unusable.
class InvalidVector : public ElipsError {
public:
    using ElipsError::ElipsError;
};

// Configuration is invalid or conflicts with persisted identity.
class ConfigError : public ElipsError {
public:
    using ElipsError::ElipsError;
};

// Requested record/vault does not exist.
class NotFound : public ElipsError {
public:
    using ElipsError::ElipsError;
};

// Persistence/IO failure.
class StorageError : public ElipsError {
public:
    using ElipsError::ElipsError;
};

}  // namespace elips

#endif  // ELIPS_DOMAIN_ERRORS_HPP
