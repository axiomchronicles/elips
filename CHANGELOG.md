# Changelog

All notable changes to ELIPS are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Parser robustness tests (`tests/fuzz/parser_robustness_test.cpp`): random
  single-byte mutation, truncation at every offset, pure-garbage input, and
  corrupted on-disk segments, asserting the WAL and snapshot readers fail fast
  and bounded rather than crashing, hanging, or over-allocating.
- `ELIPS_BUILD_FUZZERS=ON` builds a libFuzzer target for `WAL::replay()`
  (`elips_fuzz_wal`). 895k executions under ASan+UBSan produced no findings.
- `.clang-tidy` scoped to correctness, lifetime, and concurrency checks. Only the
  `WarningsAsErrors` set gates CI, so a toolchain disagreeing with its own system
  headers cannot block a merge on a false positive.
- `Vault.vacuum()` / `Database.vacuum()` (C++ and Python) to reclaim index space
  held by deleted records, plus `Vault.pending_removals` to observe how much is
  outstanding. `GraphParams::compaction_ratio` (default 0.2) controls the
  tombstone fraction that triggers automatic compaction; 0 disables it.
- `ELIPS_SANITIZE=thread|address` CMake option, building the test suite under
  ThreadSanitizer or ASan+UBSan. The full suite is clean under both. CI runs a
  ThreadSanitizer job, an ASan+UBSan job, and a job asserting that a
  no-flags Linux build does not enable the GPU engine.

### Fixed

- **GPU suballocator no longer leaks the remainder of reused blocks, and the
  free list coalesces [F9].** The best-fit path returned an oversized free block
  to the caller and dropped the leftover span entirely: not freed, not tracked,
  not reusable for the pool's lifetime. `deallocate()` also pushed blocks back
  with no merging, so alternating allocate/free at varying sizes fragmented the
  free list into pieces that never re-merged and produced spurious
  `InsufficientMemory` while plenty of total free bytes remained. Reused blocks
  now split (remainder returns to the free list), frees coalesce with physically
  adjacent spans from the same backend allocation, live block sizes are tracked
  so a double free cannot corrupt the accounting, and `bytes_available()` reports
  reachable bytes (free list + uncommitted pool headroom) instead of
  `pool_bytes - allocated`, which over-reported.

- **The GPU engine is no longer built on platforms with no GPU backend [F8].**
  `ELIPS_GPU_METAL` defaulted ON everywhere, so a plain `cmake -S . -B build` on
  Linux set `ELIPS_GPU_ENABLED`, compiled the entire 17-file `elips_gpu` library,
  and exposed `gpu_info()`/`gpu_stats()` through `elips.hpp` and the Python
  bindings with no backend behind them. The default is now `${APPLE}`, enabling
  the aggregate flag only when a backend can actually compile, and a warning
  fires if Metal is forced ON off-Apple. Backend `option()` declarations were
  duplicated between the top-level and `src/gpu_engine` CMake files — whichever
  ran first won the cache, so editing a default in one had no effect; the
  duplicates are removed and the top-level file is now the single source of
  truth. A CI job asserts the default Linux build excludes the GPU engine.

- **HNSW tombstones are bounded and no longer silently degrade recall [F5].**
  `remove()` set a flag and nothing ever reclaimed it: dead vectors and edges
  accumulated forever under insert/delete churn, and because `search()` filters
  tombstones *after* collecting a fixed-size `ef` beam, a growing dead fraction
  meant fewer than `k` live results with no error or metric. `search()` now
  scales `ef` by the total/live node ratio so the beam still yields `k` live
  hits, and the index compacts itself once tombstones pass
  `compaction_ratio` (or on an explicit `vacuum()`). *Measured:* at a 50% delete
  ratio, searches return a full `k` results with recall within 0.15 of the
  pre-delete baseline; 4,000 records churned through a 200-record live set keep
  the graph under 400 nodes instead of growing to 4,200.

- **Filtered ANN search re-probes with a wider beam instead of under-filling
  [F5, related].** A fixed `top * 20` over-fetch followed by post-filtering
  returned short result sets for selective filters. `seek()` now widens the fetch
  (4× per round, bounded by the vault size) until `top` matches are found or the
  vault is exhausted — the adaptive-`ef` gap the roadmap self-identified.

- **In-process reader/writer locking [F6].** Nothing outside `LocalTextEmbedder`
  held a lock. `LockManager` wraps a cross-process `flock` acquired once at open,
  which cannot serialize threads inside one process — so two threads sharing an
  instance raced on `HierarchicalGraphIndex`'s `data_`/`links_`/`id_to_node_` and
  on `Vault::records_`. Each `Vault` now owns a `std::shared_mutex` (readers
  share, mutators exclude) and `ElipsInstance` owns a mutex over the vault
  registry, WAL handle, and lifecycle flags. `Transaction::commit()` holds the
  instance lock for the whole batch so a concurrent writer cannot interleave
  mutations and invalidate the undo log. ADR-0008 and the transaction-engine
  internals doc previously claimed `LockManager` provided this; both corrected.
  *Behavioral change:* `Vault::records()` now returns a copy rather than a
  reference to the live map, since a reference could be mutated under a
  concurrent reader.

- **Writes after `close()` are refused instead of silently discarded [F7].**
  `close()` detached the WAL but left `Vault` objects live and writable.
  `Vault::place()` treated a null WAL as "skip logging" rather than "closed", so
  a post-close write returned a valid `RecordID`, mutated the in-memory index,
  and then vanished at process exit — silent data loss with a success return.
  `close()` now seals every vault (and any vault created afterwards); mutations
  throw `StorageError`. Reads still work.

- **WAL and checkpoint writes now reach stable storage before acknowledgement
  [F1].** `WAL::append()` called `std::ofstream::flush()`, which only pushes
  bytes into the OS page cache. Acknowledged writes were lost on OS crash or
  power loss — the exact failure mode a WAL exists to cover. The WAL now writes
  through a file descriptor and calls `fsync` (`F_FULLFSYNC` on macOS,
  `fdatasync` on Linux) before returning, and every snapshot, segment, manifest,
  and IDENTITY write is synced and published via a durable rename that also syncs
  the containing directory. *Behavioral change:* per-record write latency now
  includes a real device sync under `paranoid`/`standard` durability; `relaxed`
  is unaffected.

- **Length-prefixed reads are bounded before allocating [F2].** `get_string()`
  and `get_payload()` read a 32-bit count straight off disk and allocated for it,
  reachable from `WAL::replay()` *before* the CRC32C check that was supposed to
  reject corrupt records. A truncated or hostile log could trigger a ~4 GiB
  allocation or a 4-billion-iteration loop. Length prefixes are now validated
  against the bytes remaining in the stream (with a 64 MiB cap for non-seekable
  streams), and truncated fields raise `StorageError` instead of silently
  yielding default-constructed values.

- **`WAL::replay()` is linear in record count [F3].** Each iteration copied the
  entire remaining log tail into a fresh `std::string`, making recovery O(n·k).
  Records are now parsed in place over a non-owning stream buffer. *Measured:*
  replaying 8,000 records versus 2,000 (4× the data) now scales linearly rather
  than quadratically.

- **Transactions are atomic under I/O and read-only failure [F4].** `commit()`
  applied operations one at a time with no undo, and `rollback()` merely cleared
  an in-memory vector — so a read-only vault or a failed WAL write left the
  database holding *part* of a batch, silently. `commit()` now pre-checks that
  every target vault is writable, records each touched record's prior state, and
  restores that state in reverse order if any operation throws. The WAL brackets
  each batch with `txn_begin`/`txn_commit` markers; replay discards records
  inside an unterminated window, so a crash mid-commit also recovers to the
  pre-batch state. *Behavioral change:* `commit()` on a read-only vault now
  throws before applying anything (previously it threw partway through).
  ADR-0004 and ADR-0005 were corrected — both previously described guarantees the
  code did not deliver.
