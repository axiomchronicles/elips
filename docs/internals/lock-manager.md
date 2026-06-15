# ELIPS Lock Manager

The `LockManager` is a lightweight RAII class that enforces the single-writer contract for on-disk ELIPS databases via POSIX advisory file locking.

## Overview

ELIPS permits multiple concurrent readers but only a single writer. This contract is enforced via an advisory file lock on a `LOCK` file within the database directory. The `LockManager` acquires an exclusive, non-blocking lock at database open time and holds it for the lifetime of the `ElipsInstance`.

## Class Definition

```cpp
// include/elips/kernel/LockManager.hpp
namespace elips {

class LockConflict : public ElipsError {
public:
    using ElipsError::ElipsError;
};

class LockManager {
public:
    explicit LockManager(const std::string& lock_path);  // acquires exclusive lock
    ~LockManager();                                       // releases lock

    LockManager(const LockManager&) = delete;             // non-copyable
    LockManager& operator=(const LockManager&) = delete;
    LockManager(LockManager&&) noexcept;                  // movable
    LockManager& operator=(LockManager&&) = delete;

private:
    int fd_{-1};  // file descriptor for the lock file
};

} // namespace elips
```

## Implementation

```cpp
// src/LockManager.cpp
LockManager::LockManager(const std::string& lock_path) {
    fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        throw StorageError{"cannot open lock file: " + lock_path};
    }
    // Non-blocking exclusive lock: fail fast if another writer holds it.
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw LockConflict{"database is already open by another writer: " + lock_path};
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
```

## Locking Semantics

### Acquisition

- **Path**: The lock file is located at `<db_path>/LOCK`.
- **Open/Create**: `::open(path, O_RDWR | O_CREAT, 0644)` — opens the file for reading and writing, creating it if it doesn't exist. Permissions: owner read/write, group read, others read.
- **Lock**: `::flock(fd_, LOCK_EX | LOCK_NB)` — acquires an **exclusive** (write) lock in **non-blocking** mode.
  - `LOCK_EX`: Exclusive lock. No other process can hold any lock (shared or exclusive) on this file simultaneously.
  - `LOCK_NB`: Non-blocking. If another process holds the lock, `flock()` returns immediately with `EAGAIN`/`EWOULDBLOCK` instead of blocking.

### Release

- **Explicit**: `::flock(fd_, LOCK_UN)` releases the lock.
- **Implicit on close**: `::close(fd_)` — on POSIX, closing any file descriptor associated with the file releases all locks held by the process on that file.
- **Implicit on process exit**: The OS automatically releases all file locks when a process terminates (crash-safe).

### Error Handling

- **Open failure**: `fd_ < 0` → throws `StorageError` with the path.
- **Lock failure**: `flock() != 0` → closes the file descriptor, sets `fd_ = -1`, throws `LockConflict` with the path.

## Use in open()

```cpp
std::unique_ptr<ElipsInstance> open(const std::string& path, const Config& config) {
    // ... path validation ...

    LockManager lock{(fs::path(path) / lock_file).string()};  // single-writer

    // ... identity read/write, snapshot load, WAL replay ...

    auto instance = std::make_unique<ElipsInstance>(path, effective,
                                                    /*persistent=*/true,
                                                    std::move(lock));
    // Lock ownership is transferred to ElipsInstance
    return instance;
}
```

The `LockManager` is:
1. Created on the stack during `open()`.
2. Moved into `ElipsInstance` (via move constructor that transfers `fd_`).
3. Released when `ElipsInstance` is destroyed or `close()` is called (`lock_.reset()` triggers the destructor, which unlocks).

## RAII Guarantees

| Scenario | Lock State |
|----------|------------|
| Normal `open()` → `close()` | Lock acquired on open, released on close |
| `ElipsInstance` destructor | Lock released (destructor calls `~LockManager()` which calls `flock(LOCK_UN)`) |
| Process crash / `SIGKILL` | Lock released by OS (kernel cleans up file descriptors) |
| Exception during `open()` before `ElipsInstance` is created | Lock released (stack unwinding calls `~LockManager()`) |
| In-memory database (`":memory:"`) | No lock created (no filesystem path) |

## Cross-Platform Considerations

### Current Implementation: POSIX (macOS, Linux)

- Uses `flock()` from `<sys/file.h>` — BSD-style advisory locking.
- File descriptor based: lock follows the `fd`, not the process.

### Planned: Windows

- `LockFileEx()` with `LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY` flags.
- Windows locks are mandatory (not advisory) for the locking process, advisory for others.
- Path handling differs: `::CreateFileW()` with `GENERIC_READ | GENERIC_WRITE`.

## Multi-Reader Semantics

In the current single-writer/multi-reader model:
- A writer holds `LOCK_EX` (exclusive lock), preventing any other writers from opening.
- Readers that only need to read the database should NOT acquire a lock (they can open with an in-memory instance or a separate read-only path).

Since ELIPS is an embedded library (not a server), the "reader" process typically either:
1. Opens a separate in-memory copy of the database (no lock needed).
2. Opens the on-disk database in read-only mode (not yet implemented — a future enhancement).
3. Waits for the writer to `close()` before opening (process-level coordination).

## Lock File Location

```
<db_path>/
  LOCK          ← advisory lock file (0 bytes, used only for flock)
  IDENTITY      ← database metadata
  elips.snapshot ← latest checkpoint
  wal.log       ← write-ahead log
```

The `LOCK` file itself contains no data — it exists solely as a target for the `flock()` advisory lock. Its size is typically 0 bytes.

## Error Exception Hierarchy

```
std::runtime_error
  └── elips::ElipsError
        ├── elips::LockConflict     ← thrown when lock cannot be acquired
        ├── elips::StorageError     ← thrown when lock file cannot be opened
        ├── ...
```

In Python:
```python
try:
    db = elips.open("path/to/db")
except elips.LockConflict as e:
    print(f"Database is locked by another process: {e}")
except elips.StorageError as e:
    print(f"Storage error: {e}")
```