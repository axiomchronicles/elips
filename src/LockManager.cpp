#include "elips/kernel/LockManager.hpp"

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#include <fcntl.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#include <utility>

namespace elips {

LockManager::LockManager(const std::string& lock_path, LockMode mode)
    : mode_(mode) {
#ifdef _WIN32
    fd_ = ::_open(lock_path.c_str(), _O_RDWR | _O_CREAT | _O_BINARY, _S_IREAD | _S_IWRITE);
    if (fd_ < 0) {
        throw StorageError{"cannot open lock file: " + lock_path};
    }
    HANDLE hFile = reinterpret_cast<HANDLE>(::_get_osfhandle(fd_));
    OVERLAPPED ov{};
    DWORD flags = LOCKFILE_FAIL_IMMEDIATELY;
    if (mode_ == LockMode::exclusive) {
        flags |= LOCKFILE_EXCLUSIVE_LOCK;
    }
    if (!::LockFileEx(hFile, flags, 0, 1, 0, &ov)) {
        ::_close(fd_);
        fd_ = -1;
        if (mode_ == LockMode::exclusive) {
            throw LockConflict{"database is already open by another reader or "
                               "writer: " + lock_path};
        }
        throw LockConflict{"database is already open by a writer: " + lock_path};
    }
#else
    fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        throw StorageError{"cannot open lock file: " + lock_path};
    }
    const int op =
        (mode_ == LockMode::exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB;
    if (::flock(fd_, op) != 0) {
        ::close(fd_);
        fd_ = -1;
        if (mode_ == LockMode::exclusive) {
            throw LockConflict{"database is already open by another reader or "
                               "writer: " + lock_path};
        }
        throw LockConflict{"database is already open by a writer: " +
                           lock_path};
    }
#endif
}

LockManager::~LockManager() {
    if (fd_ >= 0) {
#ifdef _WIN32
        HANDLE hFile = reinterpret_cast<HANDLE>(::_get_osfhandle(fd_));
        OVERLAPPED ov{};
        ::UnlockFileEx(hFile, 0, 1, 0, &ov);
        ::_close(fd_);
#else
        ::flock(fd_, LOCK_UN);
        ::close(fd_);
#endif
    }
}

LockManager::LockManager(LockManager&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      mode_(std::exchange(other.mode_, LockMode::exclusive)) {}

}  // namespace elips
