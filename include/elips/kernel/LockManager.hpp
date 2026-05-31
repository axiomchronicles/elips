#ifndef ELIPS_KERNEL_LOCK_MANAGER_HPP
#define ELIPS_KERNEL_LOCK_MANAGER_HPP

#include <string>

#include "elips/domain/Errors.hpp"

namespace elips {

// Raised when a second writer tries to open a database directory already held.
class LockConflict : public ElipsError {
public:
    using ElipsError::ElipsError;
};

// RAII advisory file lock enforcing the single-writer / multi-reader contract
// across processes sharing a database directory. Uses flock() on POSIX. The
// lock is held for the lifetime of the object and released on destruction.
class LockManager {
public:
    explicit LockManager(const std::string& lock_path);  // acquires exclusive
    ~LockManager();

    LockManager(const LockManager&) = delete;
    LockManager& operator=(const LockManager&) = delete;
    LockManager(LockManager&&) noexcept;
    LockManager& operator=(LockManager&&) = delete;

private:
    int fd_{-1};
};

}  // namespace elips

#endif  // ELIPS_KERNEL_LOCK_MANAGER_HPP
