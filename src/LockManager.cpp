#include "elips/kernel/LockManager.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <utility>

namespace elips {

LockManager::LockManager(const std::string& lock_path) {
    fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        throw StorageError{"cannot open lock file: " + lock_path};
    }
    // Non-blocking exclusive lock: fail fast if another writer holds it.
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw LockConflict{"database is already open by another writer: " +
                           lock_path};
    }
}

LockManager::~LockManager() {
    if (fd_ >= 0) {
        ::flock(fd_, LOCK_UN);
        ::close(fd_);
    }
}

LockManager::LockManager(LockManager&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)) {}

}  // namespace elips
