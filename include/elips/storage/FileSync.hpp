#ifndef ELIPS_STORAGE_FILE_SYNC_HPP
#define ELIPS_STORAGE_FILE_SYNC_HPP

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#include <fcntl.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>

#include "elips/domain/Errors.hpp"

// Durability primitives. std::ostream::flush() only pushes bytes into the OS
// page cache; surviving an OS crash or power loss additionally requires the
// kernel to write them out, which is what these helpers force.
namespace elips::detail {

// Observation/fault-injection point so tests can assert the sync path actually
// runs and that a failing sync surfaces as an error instead of a silent ack.
struct SyncProbe {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<bool> fail_next{false};
    // Fail the Nth subsequent sync (1 = the very next one); 0 disables.
    std::atomic<std::uint64_t> fail_after{0};
};

inline SyncProbe& sync_probe() noexcept {
    static SyncProbe probe;
    return probe;
}

// Force the file's data (and, where the platform requires it, the drive's own
// write cache) out to stable storage. Throws StorageError on failure.
inline void sync_file_data(int fd) {
    auto& probe = sync_probe();
    probe.calls.fetch_add(1, std::memory_order_relaxed);
    if (probe.fail_next.exchange(false, std::memory_order_relaxed)) {
        throw StorageError{"fsync failed (injected)"};
    }
    if (const auto countdown = probe.fail_after.load(std::memory_order_relaxed);
        countdown > 0) {
        if (probe.fail_after.fetch_sub(1, std::memory_order_relaxed) == 1) {
            throw StorageError{"fsync failed (injected)"};
        }
    }
#ifdef _WIN32
    if (_commit(fd) != 0) {
        if (errno == EBADF) {
            return;
        }
        throw StorageError{"_commit failed"};
    }
#else
#ifdef F_FULLFSYNC
    // macOS: fsync() only reaches the drive's volatile write cache.
    if (::fcntl(fd, F_FULLFSYNC, 0) == 0) {
        return;
    }
    // Not all filesystems implement F_FULLFSYNC; fall through to fsync().
#endif
#if defined(__linux__)
    if (::fdatasync(fd) != 0) {
#else
    if (::fsync(fd) != 0) {
#endif
        throw StorageError{"fsync failed"};
    }
#endif
}

inline void sync_file_path(const std::filesystem::path& path) {
#ifdef _WIN32
    const int fd = ::_open(path.string().c_str(), _O_RDWR | _O_BINARY);
    if (fd < 0) {
        throw StorageError{"cannot open for fsync: " + path.string()};
    }
    try {
        sync_file_data(fd);
    } catch (...) {
        ::_close(fd);
        throw;
    }
    ::_close(fd);
#else
    const int fd = ::open(path.string().c_str(), O_RDONLY);
    if (fd < 0) {
        throw StorageError{"cannot open for fsync: " + path.string()};
    }
    try {
        sync_file_data(fd);
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
#endif
}

// Make a rename/create durable by syncing the containing directory. Best
// effort: some filesystems reject directory fsync, and a failure here cannot
// lose already-synced file data.
inline void sync_directory(const std::filesystem::path& dir) noexcept {
#ifdef _WIN32
    (void)dir;
#else
    const int fd = ::open(dir.string().c_str(), O_RDONLY);
    if (fd < 0) {
        return;
    }
    (void)::fsync(fd);
    ::close(fd);
#endif
}

// Publish a temp file at its final path durably: sync contents, rename, then
// sync the directory so the rename itself survives a crash.
inline void durable_rename(const std::filesystem::path& tmp,
                           const std::filesystem::path& final_path) {
    sync_file_path(tmp);
    std::filesystem::rename(tmp, final_path);
    sync_directory(final_path.parent_path());
}

}  // namespace elips::detail

#endif  // ELIPS_STORAGE_FILE_SYNC_HPP
