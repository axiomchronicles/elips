#ifndef ELIPS_DOMAIN_ERRORS_HPP
#define ELIPS_DOMAIN_ERRORS_HPP

#include <stdexcept>

namespace elips {

// Base for all ELIPS errors (E.14: purpose-designed exception types).
class ElipsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Vector dimension does not match the database/collection configuration.
class DimensionMismatch : public ElipsError {
public:
    using ElipsError::ElipsError;
};

// `*Error`-suffixed spelling (Phase 5 naming alignment), matching the Python
// alias `DimensionMismatchError = DimensionMismatch`: same type, so a throw of
// either is caught by either and pybind keeps translating it to the native
// `DimensionMismatch` Python class.
using DimensionMismatchError = DimensionMismatch;

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

// An IVF-family index was used (add/search) before train() learned its coarse
// quantizer. Derives from ConfigError: before-train use has always surfaced as
// a configuration error here, so `catch (const ConfigError&)` and the Python
// layer's ConfigError translation keep catching it, while callers that want the
// specific condition can catch NotTrainedError directly.
class NotTrainedError : public ConfigError {
public:
    using ConfigError::ConfigError;
};

// Requested record/collection does not exist.
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
