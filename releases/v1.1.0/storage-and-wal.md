# Storage Engine & Write-Ahead Log (WAL) — v1.1.0 Updates

This document detail the core persistence engine enhancements shipped in ELIPS v1.1.0, specifically covering fixes **F1**, **F2**, and **F3**.

---

## F1 — True OS/Hardware Disk Synchronization

### Problem Definition
Prior to v1.1.0, `WAL::append()` flushed records using `std::ofstream::flush()`. This operation flushes C++ user-space buffers into the Operating System page cache, but does **not** issue a hardware flush command to the physical storage device. If the host machine experienced an operating system crash, kernel panic, or sudden power outage, acknowledged vector writes remaining in page cache were permanently lost.

Furthermore, snapshot creation and manifest file updates used standard file writes without directory syncing, opening a window where renamed files could be lost or corrupted across filesystem boundary syncs.

### Implementation & Fix
In v1.1.0, `WAL` persistence was refactored to use raw file descriptors and native POSIX platform system calls:

1. **Per-Write Disk Syncing:**
   - On **Linux**: Invokes `fdatasync(fd_)` after each appended record under `paranoid` or `standard` durability levels. `fdatasync` flushes data blocks without forcing unnecessary inode metadata syncs, maximizing write throughput.
   - On **macOS**: Invokes `fcntl(fd_, F_FULLFSYNC)` after each appended record. Unlike standard `fsync()`, `F_FULLFSYNC` requests that the physical drive controller flush its internal non-volatile disk cache.

2. **Atomic Manifest & Segment Directory Sync:**
   When publishing checkpoints, snapshots, segment files, or `IDENTITY` manifests:
   - Data is written to a temporary file (`file.tmp`).
   - `fsync()` is called on `file.tmp`.
   - `std::filesystem::rename` atomically swaps `file.tmp` to `file`.
   - `fsync()` is executed on the parent directory file descriptor to ensure the directory entry update is durably committed to disk.

```
[Write Mutation] -> [WAL raw fd write] -> [fdatasync / F_FULLFSYNC] -> [Hardware Disk]
                                                                             │
[Checkpoint]     -> [Write .tmp] -> [fsync .tmp] -> [Rename] -> [fsync Dir] ─┘
```

---

## F2 — Bounded Length-Prefixed Allocation Checks

### Problem Definition
ELIPS serializes variable-length data (such as record IDs, payload strings, and vector float arrays) using a 32-bit length prefix preceding the binary payload. 

In pre-v1.1.0 releases, deserialization helper functions (`get_string()`, `get_payload()`) read the 32-bit integer prefix and immediately attempted memory allocation (`std::string::resize()` or `vector::reserve()`) **before** verifying the record CRC32C checksum. A single corrupted byte in the length field of a log or segment file could specify a length of `0xFFFFFFFF` (~4GB), triggering:
- Immediate out-of-memory (`std::bad_alloc`) exceptions.
- Massive memory allocations leading to process death by the OS OOM killer.
- Unbounded looping or hanging before reaching the CRC validation step.

### Implementation & Fix
Deserialization methods now perform strict pre-allocation boundary checks:
1. The requested payload byte length is checked against the total remaining bytes in the input stream or buffer span (`stream_size - current_offset`).
2. Non-seekable stream buffers enforce a maximum upper limit of 64 MiB per single payload field.
3. If the requested length exceeds the available payload bytes, the parser aborts immediately and raises `elips::StorageError("truncated or corrupted payload length")` without allocating memory.

---

## F3 — $O(n)$ Linear In-Place WAL Replay

### Problem Definition
During database startup, `WAL::replay()` reads existing `wal.log` files to recover mutations executed after the latest snapshot. 

In pre-v1.1.0 implementations, the replay parser read records by copying the remaining unparsed file stream into a transient `std::string` buffer on every record iteration. For a log with $N$ records and average record size $K$, this resulted in $O(N^2 \cdot K)$ memory copies, causing severe startup latency degradation for WAL files containing tens of thousands of records.

### Implementation & Fix
`WAL::replay()` has been refactored to parse records in-place using a non-owning `std::span_stream` / memory view over a single mmap'd or buffer-backed allocation:
- The log file is read or memory-mapped once into a contiguous byte buffer.
- Record headers, vectors, payloads, and CRC checksums are validated via zero-copy byte offsets.
- Recovery execution time scales strictly linearly $O(N)$ with log size.

### Benchmark Comparison (WAL Recovery Time)
| Record Count | Pre-v1.1.0 Recovery Time ($O(N^2)$) | v1.1.0 Recovery Time ($O(N)$) | Speedup |
| :--- | :--- | :--- | :--- |
| 1,000 | 4.2 ms | 0.8 ms | 5.2x |
| 10,000 | 412.0 ms | 7.1 ms | 58.0x |
| 50,000 | 9,850.0 ms | 34.5 ms | 285.5x |

---

## Code Example: Verifying WAL Integrity & Replay in Python

```python
import elips

# 1. Replay a WAL file directly without opening the database instance
try:
    entries = elips.replay_wal("/var/data/elips_db/wal.log")
    print(f"Successfully parsed {len(entries)} valid WAL entries.")
    
    for entry in entries:
        print(f"Op: {entry.op.name}, Vault: {entry.vault}, Record ID: {entry.id}")
        if entry.op == elips.WalOp.insert:
            print(f"  Vector dim: {len(entry.vector)}, Payload: {entry.data}")
            
except elips.StorageError as err:
    print(f"Corrupt tail detected and safely truncated: {err}")
```
