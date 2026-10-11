# Testing

ELIPS uses **GoogleTest v1.15.2** (obtained via `FetchContent`) with **CTest** integration via `gtest_discover_tests`. Tests are organized into categories: unit, integration, recovery, concurrency, GPU, parity, and Python.

## Test Framework

### GoogleTest

```cmake
# CMakeLists.txt (lines 84-136)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(elips_tests
    tests/unit/metrics_test.cpp
    tests/unit/index_flat_test.cpp
    # ... all test files
    tests/concurrency/multi_reader_test.cpp
)
target_link_libraries(elips_tests PRIVATE elips_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(elips_tests)
```

`gtest_discover_tests` automatically registers each `TEST()` / `TEST_F()` macro as an individual CTest test, enabling both CTest filtering and `--gtest_filter` usage.

## Running Tests

### All tests

```bash
ctest --test-dir build --output-on-failure
```

### By category

```bash
# All unit tests
ctest --test-dir build -R "unit"

# Integration tests
ctest --test-dir build -R "Integration"

# Recovery tests
ctest --test-dir build -R "Recovery"

# Concurrency tests
ctest --test-dir build -R "Concurrency"

# GPU tests
ctest --test-dir build -R "Gpu\|CpuGpu"

# EQL tests
ctest --test-dir build -R "Eql"
```

### GoogleTest filtering

```bash
# Run specific test suite
./build/elips_tests --gtest_filter='IndexFlatTest.*'

# Run specific test case
./build/elips_tests --gtest_filter='MetricsTest.CosineIdenticalNormalizedVectorsZeroDistance'

# Run across multiple patterns
./build/elips_tests --gtest_filter='*Edge*:*Recall*'

# List all registered tests
./build/elips_tests --gtest_list_tests

# Run disabled tests
./build/elips_tests --gtest_also_run_disabled_tests
```

### Python tests

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DELIPS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DELIPS_BUILD_CLI=OFF
cmake --build build --target elips_pymodule -j
PYTHONPATH=bindings/python python3 tests/python/test_bindings.py
```

## Test Organization

```
tests/
├── unit/                          # Isolated component tests
│   ├── metrics_test.cpp           # Distance functions
│   ├── index_flat_test.cpp        # IndexFlat (brute-force)
│   ├── index_flat_edge_cases_test.cpp  # IndexFlat edge cases
│   ├── record_id_test.cpp         # UUIDv7 RecordID
│   ├── vector_test.cpp            # Vector value type
│   ├── config_test.cpp            # Config builder
│   ├── eql_test.cpp               # EQL parsing and execution
│   ├── eql_edge_test.cpp          # EQL edge cases
│   ├── filter_edge_test.cpp       # Filter predicate engine
│   ├── scan_test.cpp              # Collection::scan()
│   ├── durability_test.cpp        # Durability levels
│   ├── hnsw_recall_test.cpp       # HNSW recall vs. ground truth
│   ├── hnsw_edge_cases_test.cpp   # HNSW edge cases
│   └── gpu/                       # GPU unit tests (conditional)
│       ├── GpuDeviceManagerTest.cpp
│       ├── GpuMemoryManagerTest.cpp
│       └── QueryBatcherTest.cpp
├── integration/                   # Cross-component tests
│   ├── roundtrip_test.cpp         # add → get → search → remove
│   ├── transaction_filter_test.cpp # Transactions + filtering
│   └── gpu/                       # GPU integration tests (conditional)
│       └── MetalBackendTest.cpp
├── recovery/                      # Crash recovery and WAL
│   └── wal_recovery_test.cpp
├── concurrency/                   # Multi-thread correctness
│   ├── single_writer_test.cpp
│   └── multi_reader_test.cpp
├── parity/                        # CPU/GPU result parity (conditional)
│   └── CpuGpuRecallParityTest.cpp
└── python/                        # Python binding validation
    └── test_bindings.py
```

GPU and parity tests are conditionally compiled and only run when `ELIPS_GPU_ENABLED` is set and a GPU backend is available.

## Complete Test Catalog

### Unit Tests

#### `tests/unit/metrics_test.cpp` — 5 tests

| Test | Description |
|---|---|
| `CosineIdenticalNormalizedVectorsZeroDistance` | Cosine distance of identical unit vectors is 0.0 |
| `CosineOrthogonalNormalizedVectorsDistanceOne` | Orthogonal unit vectors have cosine distance 1.0 |
| `EuclideanMatchesManualComputation` | sqrt((3-0)² + (4-0)²) = 5.0 |
| `DotProductIsNegatedForAscendingSort` | dot=6 → distance=-6 so larger dots sort first |
| `OnlyCosineRequiresNormalization` | `requires_normalization()` returns true only for cosine |

#### `tests/unit/index_flat_test.cpp` — 4 tests

| Test | Description |
|---|---|
| `ReturnsNearestFirstAscending` | Search returns hits sorted ascending by distance |
| `TopKCapsResultCount` | k=3 returns exactly 3 results when 5 are inserted |
| `RemoveDropsRecordFromResults` | `remove()` excludes the deleted record from search |
| `TypeNameIsExact` | `type_name()` returns `"exact"` |

#### `tests/unit/index_flat_edge_cases_test.cpp` — 9 tests

| Test | Description |
|---|---|
| `EmptyIndexSearchReturnsEmpty` | Search on empty index returns no hits, size=0 |
| `RemoveNonExistentIsNoop` | Removing a non-existent id does not throw |
| `ReInsertSameId` | Re-inserting the same id appends (does not overwrite) |
| `SearchAsksZeroResults` | k=0 returns empty result |
| `SearchAsksMoreThanInserted` | k > count returns min(count, k) results |
| `HighDimensionalVector` | Works with 1024-dimensional vectors |
| `AllMetricsWork` | Cosine, euclidean, dot_product all function correctly |
| `RemoveThenReinsertAtEnd` | Remove + subsequent insert maintains consistency |
| `TypeNameIsCorrectAcrossMetrics` | type_name is `"exact"` regardless of metric |

#### `tests/unit/record_id_test.cpp` — 4 tests

| Test | Description |
|---|---|
| `GenerateProducesVersion7Variant10` | UUIDv7: byte[6] high nibble = 0x7, byte[8] high 2 bits = 0b10 |
| `StringRoundTrip` | `to_string()` → `from_string()` produces identical RecordID |
| `ToStringIsCanonicalHyphenatedForm` | Standard 8-4-4-4-12 UUID format with hyphens |
| `TimePrefixIsMonotonicAcrossTime` | UUIDs generated 2ms apart have increasing time prefixes |

#### `tests/unit/vector_test.cpp` — 11 tests

| Test | Description |
|---|---|
| `DefaultConstructedIsEmpty` | Vector{} has dimension 0, empty=true |
| `ConstructedWithValues` | Vector{{1,2,3}} stores and reports correct values |
| `MagnitudeComputation` | sqrt(3² + 4²) = 5.0 |
| `ZeroVectorMagnitude` | Zero vector has magnitude 0.0 |
| `NormalizedIsUnitLength` | `normalized()` produces vector with magnitude 1.0 |
| `NormalizedZeroVectorIsUnchanged` | Zero vector normalization returns zero values |
| `SingleElementNormalization` | [7.0] normalizes to [1.0] |
| `LargeVectorNormalization` | 1024-element unit vector normalizes correctly |
| `NegativeValuesNormalization` | [-3, -4] normalizes to [-0.6, -0.8] |
| `EmptyVectorNormalization` | Normalizing empty vector returns empty vector |
| `DimensionMatchesSize` | dimension() equals vector size for various dimensions |

#### `tests/unit/config_test.cpp` — 11 tests

| Test | Description |
|---|---|
| `DefaultValuesAreSane` | Default dimension=0, metric=cosine, index=graph, durability=standard |
| `FluentBuilderChainSetsAllValues` | Chained `.dimension().metric().index().durability()` works |
| `GraphParamsDefaults` | Default M=16, ef_construction=200, ef_search=50 |
| `CustomGraphParams` | GraphParams{32,400,100} stores correctly |
| `AllDurabilityLevels` | paranoid, standard, relaxed, ephemeral all settable |
| `IndexTypeValues` | graph and exact both settable |
| `MetricToFromStringRoundTrip` | to_string ↔ from_string for all metrics |
| `MetricFromStringCaseSensitive` | "cosine", "euclidean", "dot_product" parse correctly |
| `ToStringValues` | to_string returns expected lowercase strings |
| `GraphParamsConstructor` | Default and custom constructor values |
| `ConfigIsCopyable` | Config copy preserves all field values |

#### `tests/unit/eql_test.cpp` — 6 tests

| Test | Description |
|---|---|
| `ParsesSearchStatement` | Parser extracts collection, binding, k, projection from seek |
| `PlaceThenSeekViaEql` | Place 2 vectors, seek returns correct nearest match |
| `FilteredScanViaEql` | `scan where tier = "pro"` returns 2 of 3 rows |
| `FetchAndEraseViaEql` | Place → fetch by id → erase → verify count=0 |
| `InClauseAndBoolean` | `where country in ["US","GB"] and ok = true` filters correctly |
| `MalformedQueryThrowsParseError` | `"seek docs nearest"` throws ParseError |

#### `tests/unit/eql_edge_test.cpp` — 12 tests

| Test | Description |
|---|---|
| `ParseWithComments` | `#` line comments are ignored |
| `ParseEmptyStringThrows` | Empty/whitespace input throws ParseError |
| `TokenizeProducesAtLeastEndToken` | Empty input produces single end token |
| `TokenizeStringWithEscapes` | Quoted strings with escapes tokenize correctly |
| `TokenizePunctuationTokens` | >=, <=, !=, =, <, >, and brackets are punct tokens |
| `ComplexQueryParsesCorrectly` | Seek with bindings, filter, projection works end-to-end |
| `FetchNonExistentReturnsEmpty` | Fetch of unknown UUID returns empty result |
| `ScanWithLimitAndOffsetParses` | `scan offset 2 limit 3` returns exactly 3 results |
| `ThresholdSearchParses` | `threshold 0.5` limits results to close neighbors |
| `ProjectFieldsReturnSubset` | `project a, c` returns only those data fields |
| `MixedBooleanFilter` | `and`, `or`, `not` in scan filters work correctly |
| `EmptyBindingsIsSafe` | `seek nearest [1.0, 0.0] yield` with empty vault works |

#### `tests/unit/filter_edge_test.cpp` — 12 tests

| Test | Description |
|---|---|
| `DefaultFilterMatchesEverything` | Empty Filter matches any payload |
| `EqualityComparisonTypes` | equals works with int64, string, bool, double |
| `InequalityComparisons` | gt, ge, lt, le comparisons on int64 |
| `NotOperator` | `not_(filter)` inverts match result |
| `AndOrComposition` | `a.and_(b)`, `a.or_(b)` boolean logic |
| `DeeplyNestedAndOr` | `(a and b) or (c and d)` nested composition |
| `ContainsSubstring` | `contains("hello")` matches "hello world" |
| `OneOfSetMembership` | `one_of([a,b,c])` set membership |
| `EmptyOneOfNeverMatches` | `one_of([])` matches nothing |
| `MissingFieldDoesNotMatch` | Predicate on non-existent field returns false |
| `ChainedBuilderProducesAnd` | `.field("a").equals(1).field("b").gt(0)` is AND |
| `NotEqualOperator` | `not_equals(0)` matches only non-zero values |

#### `tests/unit/scan_test.cpp` — 8 tests

| Test | Description |
|---|---|
| `EmptyVaultScanReturnsEmpty` | Scan on empty vault returns 0 rows |
| `ScanReturnsAllRecords` | 10 inserted → scan returns 10 |
| `ScanWithOffsetReturnsFewer` | offset=3 on 5 records returns 2 |
| `ScanWithLimitTruncates` | limit=3 on 10 records returns 3 |
| `ScanWithFilteredSearch` | Filter by tag returns correct subsets |
| `ScanOffsetBeyondSizeReturnsEmpty` | offset > count returns empty |
| `LargeLimitReturnsAll` | limit=999999 on 50 records returns 50 |
| `ScanWithMixedFilterAndLimitOffset` | Filter + offset + limit combined correctly |

#### `tests/unit/durability_test.cpp` — 7 tests

| Test | Description |
|---|---|
| `ParanoidSurvivesCrash` | `abandon()` (simulated crash) recovers writes |
| `StandardPersistsOnClose` | `close()` checkpoints and data survives reopen |
| `RelaxedPersistsOnCheckpoint` | `checkpoint()` + `abandon()` recovers |
| `EphemeralDoesNotPersist` | `:memory:` databases have no persistence |
| `MultipleRecordsSurviveStandard` | 100 records persist through checkpoint + reopen |
| `MultipleVaultsSurviveCheckpoint` | Two vaults both persist through checkpoint |
| `DeletionSurvivesCheckpoint` | Erased records remain deleted after checkpoint + reopen |

#### `tests/unit/hnsw_recall_test.cpp` — 2 tests

| Test | Description |
|---|---|
| `RecallAtTenIsHighVsExactGroundTruth` | HNSW recall@10 ≥ 90% vs ExactIndex on 2000x64 data |
| `SoftRemoveExcludesFromResults` | Removed nodes do not appear in search results |

#### `tests/unit/hnsw_edge_cases_test.cpp` — 10 tests

| Test | Description |
|---|---|
| `EmptyIndexSearchReturnsEmpty` | Search on empty graph returns no hits |
| `SingleElementInsertAndSearch` | Insert one vector, search finds it |
| `DuplicateIdsAccumulate` | Re-inserting same id increments count |
| `RemoveThenSearchExcludesDeleted` | 3 nodes, remove middle, it's excluded from results |
| `RemoveAllThenSearchEmpty` | Removing all 5 nodes leaves index empty |
| `SearchRequestsMoreThanInserted` | k=100 on 5 nodes returns 5 results |
| `HighDimensionalSingleInsert` | Single 768-dim vector inserts and searches correctly |
| `CustomGraphParamsAffectSearch` | Tight params (M=4, ef=50, ef_search=20) still works |
| `TypeNameIsGraph` | `type_name()` returns `"graph"` |
| `AllMetricsWork` | Cosine, euclidean, dot_product all function |
| `InsertionOrderDoesNotAffectSize` | 100 inserts in known order → size is exactly 100 |

### GPU Unit Tests (conditional on `ELIPS_GPU_ENABLED`)

#### `tests/unit/gpu/GpuDeviceManagerTest.cpp` — 4 tests

| Test | Description |
|---|---|
| `probe_returns_results_on_supported_system` | Metal backend returns devices on Apple hardware |
| `selector_chooses_best_available` | `GpuPolicy::Auto` selects a backend when GPUs exist |
| `policy_cpu_only_returns_no_backend` | `GpuPolicy::CpuOnly` returns nullopt |
| `can_fit_index_estimation` | Memory estimation for CAGRA graph index |

#### `tests/unit/gpu/GpuMemoryManagerTest.cpp` — 4 tests (1 disabled)

| Test | Description |
|---|---|
| `allocate_and_free_succeed` | Allocate 4096 bytes → non-null device pointer → free |
| `double_free_is_safe` | Freeing an empty GpuBuffer is safe |
| `peak_bytes_tracking` | `bytes_used()` and `peak_bytes_used()` track correctly |
| `pinned_allocation_works` | Pinned (host-accessible) memory allocation |
| `acquire_and_release` | **Disabled** — skipped due to Metal backend issue |

#### `tests/unit/gpu/QueryBatcherTest.cpp` — 3 tests

| Test | Description |
|---|---|
| `single_query_completes` | Single query → single batch → result returned |
| `batch_coalesces_concurrent_queries` | 10 concurrent queries coalesced into ≤5 batches |
| `stats_are_accurate` | `queries_coalesced`, `kernel_launches`, `avg_batch_size` tracked |

### Integration Tests

#### `tests/integration/roundtrip_test.cpp` — 8 tests

| Test | Description |
|---|---|
| `InMemoryPlaceAndSeek` | Place 2 vectors, seek returns correct nearest |
| `PersistenceRoundtripAcrossReopen` | Place → checkpoint → reopen → fetch/seek all correct |
| `DimensionMismatchThrows` | 2-dim vector in 3-dim vault throws DimensionMismatch |
| `NonFiniteVectorRejected` | Inf values throw InvalidVector |
| `ReopenWithConflictingDimensionThrows` | Reopening with wrong dimension throws ConfigError |
| `EraseRemovesRecord` | Erase removes from fetch, search, and count |
| `AutoCheckpointOnDestruction` | Destructor checkpoints without explicit call |

#### `tests/integration/transaction_filter_test.cpp` — 6 tests

| Test | Description |
|---|---|
| `FilteredSeekRestrictsByMetadata` | category=tech AND year>=2020 returns 1 record |
| `ScanWithFilterAndLimit` | Filtered scan + limit correctness |
| `OrAndNotComposition` | Boolean filter composition with or, not |
| `CommitAppliesAllWrites` | Transaction commit applies both inserts |
| `UncommittedTransactionRollsBack` | Transaction destructor without commit = rollback |
| `InvalidVectorRejectedAtBufferTime` | Transaction rejects dimension mismatch at enqueue |

### Integration GPU Tests (conditional)

#### `tests/integration/gpu/MetalBackendTest.cpp` — 4 tests

| Test | Description |
|---|---|
| `initialize_succeeds` | GPU backend reports available |
| `allocate_upload_download_roundtrip` | Host→GPU→Host data transfer preserves values |
| `cosine_distance_matches_cpu` | GPU cosine distance equals CPU scalar result |
| `brute_force_index_search` | GpuBruteForceIndex builds and searches correctly |

### Recovery Tests

#### `tests/recovery/wal_recovery_test.cpp` — 3 tests

| Test | Description |
|---|---|
| `WalReplayRecoversUncheckpointedWrites` | Crash (abandon) without checkpoint → reopen recovers 2 records |
| `CorruptWalTailIsTruncatedNotFatal` | Garbage bytes appended to WAL → recovery truncates cleanly, retains valid records |
| `WalAppendAndReplayRoundTrip` | Direct WAL insert+erase → replay returns both entries |

### Concurrency Tests

#### `tests/concurrency/single_writer_test.cpp` — 1 test

| Test | Description |
|---|---|
| `SecondWriterFailsWithLockConflict` | Opening same directory twice throws LockConflict; first close allows reopening |

#### `tests/concurrency/multi_reader_test.cpp` — 4 tests

| Test | Description |
|---|---|
| `MultipleReadersCanOpenSameDatabase` | 4 concurrent readers all successfully read |
| `ConcurrentReadsOnInMemory` | 8 threads × 50 seeks = 400 queries on in-memory db |
| `SequentialTransactionsWorkReliably` | 100 sequential txn commits = 100 records |
| `ConcurrentReadAttemptsOnPersistentDb` | 4 threads attempt reads on persistent db |

### Parity Tests (conditional)

#### `tests/parity/CpuGpuRecallParityTest.cpp` — 1 test

| Test | Description |
|---|---|
| `brute_force_recall_perfect` | GPU brute-force recall@5 ≥ 95% vs CPU ExactIndex on 100×8 data |

### Python Tests

#### `tests/python/test_bindings.py` — 18 test functions

| Test | Description |
|---|---|
| `test_exceptions` | All 8 exception types importable, raised, have correct hierarchy |
| `test_enums` | All core and GPU enum values exist and are distinct |
| `test_utilities` | distance(), requires_normalization(), metric_to/from_string() with string/enum overloads |
| `test_eql_tooling` | tokenize_eql(), validate_eql() for valid and invalid EQL |
| `test_config` | Config builder, property getters, open_with_config() |
| `test_database_crud` | place, seek, fetch, scan, erase, count, info |
| `test_place_many` | Batch place_many with and without explicit IDs |
| `test_filtered_search` | Filter equals, gte, or_, not_, one_of, contains |
| `test_transaction` | Context manager commit, exception rollback, manual commit/rollback |
| `test_database_context_manager` | `with elips.open(...)` checkpoints and releases correctly |
| `test_eql_query` | Place, seek, fetch, scan, erase via EQL query() |
| `test_gpu_config` | GpuConfig, GpuIndexBuildParams, IvfPqBuildParams property setting |
| `test_gpu_device_info` | GpuDeviceInfo has all expected attributes |
| `test_thread_safety_python` | 8 threads concurrent seek, all return 10 results |
| `test_edge_cases` | Empty vector, large dim, zero top-k, negative limit, Unicode metadata |
| `test_memory_leak_check` | 50 cycles of open/insert/search/close with gc |
| `test_type_stubs` | `py.typed` and `_core.pyi` stub files present |
| `test_parity_cpp_vs_python` | Python results match expected C++ behavior |

## Test Count Summary

| Category | Files | Tests |
|---|---|---|
| Unit (core) | 13 | ~101 |
| Unit (GPU) | 3 | 11 |
| Integration | 2 | 14 |
| Integration (GPU) | 1 | 4 |
| Recovery | 1 | 3 |
| Concurrency | 2 | 5 |
| Parity (GPU) | 1 | 1 |
| Python | 1 | 18 |
| **Total** | **24** | **~157** |

Coverage areas: distance metrics, vector normalization, UUIDv7 generation, exact nearest-neighbor search, ANN recall, index edge cases, configuration validation, EQL lexing/parsing/execution, filter predicate evaluation, scan pagination, durability across all 4 levels, WAL replay and corruption recovery, single-writer locking, multi-reader concurrency, transaction commit/rollback, GPU device probing, GPU memory management, GPU batching, Metal backend roundtrip, CPU/GPU recall parity, and full Python binding surface.

## CI Workflows

### `build.yml` (PRs and pushes to main)

| OS | Build Type | Steps |
|---|---|---|
| ubuntu-latest | Debug | configure → build → ctest |
| ubuntu-latest | Release | configure → build → ctest |
| macos-latest | Debug | configure → build → ctest |
| macos-latest | Release | configure → build → ctest |

Uses Ninja generator. All 4 jobs run independently (fail-fast: false). `--output-on-failure` ensures test failure output is visible in CI logs.

### `test.yml` (pushes to main only)

| Job | Platform | Details |
|---|---|---|
| ASan + UBSan | ubuntu-latest | Debug build with `-fsanitize=address,undefined -fno-omit-frame-pointer`, then ctest |
| Python bindings | ubuntu-latest | Release build with `-DELIPS_ENABLE_PYTHON=ON`, builds `elips_pymodule`, runs `examples/python/01_getting_started.py` smoke test |

## Writing New Tests

1. Place tests in the appropriate directory:
   - Isolated unit logic → `tests/unit/`
   - Cross-component integration → `tests/integration/`
   - Crash recovery → `tests/recovery/`
   - Thread safety → `tests/concurrency/`
   - CPU/GPU comparison → `tests/parity/`
   - Python bindings → `tests/python/`

2. Add the file to `CMakeLists.txt` in the `elips_tests` source list.

3. Follow existing test patterns:
   - Use anonymous namespace for file-local helpers
   - Use `TEST()` for stateless tests, `TEST_F()` when you need a fixture with `SetUp()`/`TearDown()`
   - Database tests that touch the filesystem should use `std::filesystem::temp_directory_path()` with random directory names, cleaned up in `TearDown()`
   - GPU tests should `GTEST_SKIP()` when no GPU is available

4. Run the complete test suite before submitting.