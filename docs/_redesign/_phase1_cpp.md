# ELIPS C++ Core Inventory — Phase 1A

Scope: the C++ **core**, excluding the GPU engine (`src/gpu_engine`, `include/elips/gpu_engine`, `tests/*/gpu`, `benchmarks/gpu` — covered by a separate agent). Covers `include/elips/{Config,domain,index_engine,kernel,metadata,quant_engine,query_engine,storage,text_engine,vector_engine,elips.hpp}`, the matching `src/`, `cli/`, and `benchmarks/BenchMain.cpp`.

Method: bodies were read, not inferred from names. **"complete"** = real body AND an in-scope test exercises it (test cited). **"untested"** = real body, no in-scope test. **stub/partial/TODO** = empty, throwing "not implemented", or incomplete. All paths repo-relative to `/Users/kuroyami/TuboxLabProject/ellips`.

Test wiring confirmed in `CMakeLists.txt:156-184`: all 27 non-GPU test files compile into one `elips_tests` gtest binary via `gtest_discover_tests` (`CMakeLists.txt:219`); `tests/fuzz/fuzz_wal_replay.cpp` is a separate `elips_fuzz_wal` target (`CMakeLists.txt:231`).

Global result: **No `TODO`/`FIXME`/throwing-"not implemented"/empty-stub exists anywhere in the in-scope sources.** Every method has a real body. The single marked-unimplemented feature is `QuantParams::refine_fraction` (reranking), and it points at a non-existent ADR. All "gaps" below are therefore *real code that is simply untested in scope* or *shipped behavior that diverges from a comment/plan*, not missing implementations.

---

## domain

Header-only value types plus `RecordID` (the one with a `.cpp`).

### Errors (`include/elips/domain/Errors.hpp`)
Exception hierarchy, all inherit `std::runtime_error` ctors, no custom logic.
- `elips::ElipsError` (`Errors.hpp:9`), `DimensionMismatch` (`:15`), `InvalidVector` (`:21`), `ConfigError` (`:27`), `NotFound` (`:33`), `StorageError` (`:39`).
- `StorageError` — **complete**: thrown + asserted in `m1_durability_atomicity_test.cpp:67,100,108,175` and `m2_thread_safety_test.cpp:198-226`. Others are reachable but their throw-paths are mostly **untested** in-scope (`DimensionMismatch`, `InvalidVector`, `NotFound`, and `ConfigError`'s throw path).

### Expected (`include/elips/domain/Expected.hpp`)
- `std::expected`/`std::unexpected` polyfill (`:17-88`), compiled in **only** when `__cpp_lib_expected` is absent (`:9`). On this C++23 repo it is inert/dead. **untested**. No in-scope API returns `std::expected` except the two `IndexTransferPort` snapshot methods.

### Record (`include/elips/domain/Record.hpp`)
- `MetaValue = variant<int64_t,double,bool,string>` (`:17`), tag order 0..3 is the on-disk encoding. **complete** (round-tripped `wal_recovery_test.cpp:87,165`).
- `Payload = map<string,MetaValue>` (`:20`). **complete**.
- `struct DocumentAttachment{text;uri;mime_type="text/plain"}` (`:22`), `struct ChunkInfo{document_key;ordinal;char_start;char_end}` (`:28`), `struct EmbeddingLineage{provider;model;revision;Payload attributes}` (`:35`). All **complete** (`wal_recovery_test.cpp:49-64,79-88,158-166`).
- `struct Record{id;vector;payload;optional document/chunk/lineage;codes;codec}` (`:50`). In a quantized vault `codes`+`codec` are the owned truth; `vector` is a reconstruction (comment `:42-49`). **complete** as a recovered value (`wal_recovery_test.cpp:76-88`). `Record::approximate()` (`:63`, returns `codec!=none`) **untested**.

### RecordID (`include/elips/domain/RecordID.hpp`, `src/RecordID.cpp`)
UUIDv7 128-bit identity (`std::array<uint8_t,16>`).
- `generate()` (`.cpp:46`) — 48-bit big-endian ms prefix + 10 random bytes (thread-local mt19937_64 seeded clock^random_device^thread-id), version-7 nibble + variant-10 bits. **complete** (`record_id_test.cpp:13-18,35-41`, monotonic time prefix).
- `to_string()` (`.cpp:74`) canonical 8-4-4-4-12 hex — **complete** (`record_id_test.cpp:21-33`).
- `from_string()` (`.cpp:89`) strips hyphens, requires 32 hex digits else throws `ConfigError` (`.cpp:98`). Happy path **complete** (`record_id_test.cpp:20`); error path **untested**.
- `operator==`/`<=>` defaulted (lexicographic = time order) — **complete**. `std::hash<RecordID>` FNV-1a (`:36`) — **untested**.

### SearchResult (`include/elips/domain/SearchResult.hpp`)
- `struct SearchResult{id;distance=0;data;optional document/chunk/lineage;codec}` (`:15`), distance smaller=closer, ordering-normalized across metrics. The type flows through `seek()` and is **complete** at the count/id level (`multi_reader_test.cpp:74`, `m2_thread_safety_test.cpp:62`), but the non-id fields (`distance`,`data`,`codec`,`approximate()`) are **untested** in scope.

### Vector (`include/elips/domain/Vector.hpp`)
Owned fp32 vector. `Vector()`/`Vector(vector<float>)` (`:16`), `values()` span (`:19`), `dimension()` u16 cast (`:20`), `empty()` (`:23`), `magnitude()` sum-of-squares+sqrt (`:25`), `normalized()` L2 copy, zero-vector returned unchanged (`:34-44`). All **complete** (`vector_test.cpp:13-86`).

---

## index_engine

Verdict up front: **`HierarchicalGraphIndex` is a real from-scratch HNSW**, not a brute-force fallback. `ExactIndex` is the genuine brute-force oracle.

### IndexPort (`include/elips/index_engine/IndexPort.hpp:16`)
Abstract port / DIP seam. `using Hit = std::pair<RecordID,float>` (id, ascending distance, `:18`). Non-copyable/non-movable. Pure virtuals: `insert(RecordID, span<const float>)` (`:28`), `remove(RecordID)` (`:29`), `search(span<const float>, k) const -> vector<Hit>` (`:32`), `size() const` (`:35`), `type_name() const` (`:36`). Overridable-with-default: `vacuum(){}` (`:40`), `pending_removals(){return 0;}` (`:43`). Concrete implementers: **`ExactIndex`**, **`HierarchicalGraphIndex`** (both also implement `IndexTransferPort`). Comment `:14-15` ("HNSW plugs in … later") is **stale** — HNSW already ships. **complete** (via both concrete classes).

### IndexTransferPort (`IndexTransferPort.hpp:11`)
Abstract: `export_snapshot() const` (`:15`), `import_snapshot(const IndexSnapshot&)` (`:18`), both `std::expected<…,std::string>`. Implemented by both indexes. (The only in-scope API using the `std::expected` polyfill.)

### IndexSnapshot (`IndexSnapshot.hpp`)
Plain data structs: `enum class IndexSnapshotKind{unknown,exact,graph,gpu_*}` (`:14`), `IvfSnapshot` (`:26`, GPU/IVF only), `PqSnapshot` (`:36`, `pq_dim` doubles as code width in bytes), `IndexSnapshot` (`:48`, canonical CPU↔GPU transfer format, fp32 `vectors` = source of truth). **untested** in scope — no snapshot test among the in-scope files.

### ExactIndex (`ExactIndex.hpp:26`, `src/ExactIndex.cpp`)
`ExactIndex(Metric, uint16_t dim, QuantizerPtr={})`. Brute-force linear scan; dual-mode fp32 (`data_`) or PQ/codec codes (`codes_`, `row_width_=code_bytes()`).
- `insert` (`.cpp:24`) — validates dim (throws `DimensionMismatch` `.cpp:17`), append-only (re-insert duplicates id). **complete** (`exact_index_test.cpp:18,61`, `exact_edge_cases_test.cpp:34`).
- `remove` (`.cpp:37`) — linear find+erase, missing id is silent no-op. **complete** (`exact_index_test.cpp:43`, `exact_edge_cases_test.cpp:27,78`).
- `search` (`.cpp:55`) — per-row `distance()` (uncompressed) or `make_lut`+`lut_distance` (compressed), `partial_sort` ascending, truncate to k. **complete** (`exact_index_test.cpp:18,34`, `exact_edge_cases_test.cpp:20,42,49,57,67`).
- `size`/`type_name` ("exact"/"exact_quantized") — **complete**.
- `export_snapshot` (`.cpp:89`)/`import_snapshot` (`.cpp:120`) — real; import re-encodes from the fp32 payload and deliberately ignores `snapshot.pq` (comment `.cpp:140-142`). **untested** in scope.
- Quantized/compressed path (`compressed()`-true, LUT) — real but **untested** (no quantizer-backed ExactIndex test in scope).

### HierarchicalGraphIndex (`HierarchicalGraphIndex.hpp:39`, `src/HierarchicalGraphIndex.cpp`)
Genuine HNSW. ctor (`.cpp:17`) sets `level_mult_=1/ln(max(M,2))`, seeds RNG. Confirmed by body:
- `random_level()` (`.cpp:77`) = standard exponential-decay level draw.
- `search_layer` (`.cpp:86`) — real beam search: min-heap frontier + bounded max-heap of size `ef`, visited set, early termination (`.cpp:106`).
- `insert`→`insert_unchecked` (`.cpp:190/195`) — level assign, greedy `ef=1` descent through upper layers (`.cpp:222`), `search_layer(ef_construction)`+`connect` per level, entry-point promotion; layer-0 uses `max_connections*2` links.
- `connect` (`.cpp:134`) — **HNSW Algorithm-4 diversity heuristic** (keep candidate only if closer to new node than to any selected neighbor), reverse edges + prune neighbor to closest `max_links`.
- `remove` (`.cpp:246`) — **soft-delete tombstone** (`deleted_[node]`), node stays as routing waypoint, auto-triggers `vacuum()` at `deleted/total ≥ compaction_ratio` (`.cpp:257`).
- `search` (`.cpp:339`) — greedy descent then **adaptive beam** `ef=ef_search*total/live` when tombstones exist (`.cpp:356-364`), filters deleted, returns k live hits.
- `vacuum` (`.cpp:264`) — rebuilds from live rows, carries codes verbatim when compressed (no re-quantization loss).

CRUD+search+vacuum **complete and well-tested**: recall@10 ≥ 0.90 vs ExactIndex oracle (`hnsw_recall_test.cpp:23,67-68`), adaptive-ef under churn (`m3_index_health_test.cpp:92`), vacuum/compaction (`m3_index_health_test.cpp:26,63`), edges (`hnsw_edge_cases_test.cpp:26,51,65,79,109,114`). `type_name` = "graph"/"graph_quantized". `export_snapshot`/`import_snapshot` (`.cpp:381/426`) and the quantized/compressed path are real but **untested** in scope.
- Private `Probe` (query-bound distance evaluator, rvalue overload `=delete`d at `.hpp:86` to prevent a dangling span) — real.
- **Stale/misleading**: two comments (`.hpp` soft-delete note, `.cpp:269-270`) reference an `unlink()` method that **does not exist**; `remove()` tombstones in place and `vacuum()` rebuilds.

### IndexFactory (`IndexFactory.hpp:20`, `src/IndexFactory.cpp`)
`make_index(const Config&, uint16_t dim [, gpu::GpuPort*])`. `make_cpu_index` (`.cpp:15`) switches on `config.index()`: exact→ExactIndex, graph→HierarchicalGraphIndex, default→ExactIndex; passes `config.quantizer()`. GPU block behind `ELIPS_GPU_ENABLED`. Real code, but **untested at factory level** (no in-scope test calls `make_index`; tests construct concrete classes or go through `elips::open`).

---

## quant_engine

Namespace `elips::quant`. **All codecs are genuinely implemented end-to-end — no stubs.** The one reserved/unimplemented knob is `refine_fraction`.

### QuantParams (`include/elips/quant_engine/QuantParams.hpp`)
- `enum class CodecId:uint8_t{none=0,pq=1,opq=2,sq8=3}` (`:21`), persisted on disk — never renumber. **complete** (`quantizer_test.cpp:426`, every value `m5_quantization_persistence_test.cpp:548`).
- `to_string(CodecId)` (def `Quantizer.cpp:13`), `codec_from_string()` (`Quantizer.cpp:27`, throws `ConfigError` on null/unknown, accepts "sq8" and "sq"). **complete** (`quantizer_test.cpp:429-435`).
- `struct QuantParams{codec;pq_dim=0(auto);pq_bits=8;train_iters=10;opq_iters=4;refine_fraction=0}` (`:33`). **partial** — `refine_fraction` is a declared-but-unused reserved field.
- **The one TODO in the whole scope**: `QuantParams.hpp:45-47` — `refine_fraction` "Reserved: … exact reranking. Not implemented; see ADR-0011." Confirmed unreferenced in any source. **ADR-0011 does not exist** (docs/adr only goes to 0010).

### Quantizer base (`Quantizer.hpp:30`, `src/quant_engine/Quantizer.cpp`)
Abstract base: pure virtuals `codec/code_bytes/dimension/metric/encode/decode/make_lut/lut_distance/serialize`; non-virtual convenience `encode(span)->vector` / `decode(span)->vector`. **complete** (exercised through both derived codecs). Free functions:
- `train(QuantParams, Metric, dim, rows, n)` (`.cpp:98`) — validates, rejects none/n==0, dispatches sq8→ScalarQuantizer else ProductQuantizer, opq only when codec==opq. **complete** (happy + error paths `quantizer_test.cpp:304-337`).
- `load(istream&)` (`.cpp:126`) — reads CodecId byte, dispatches deserialize, throws `StorageError` on truncation/unknown. **complete** (`quantizer_test.cpp:246,277`).
- `choose_pq_dim(dim,configured)` (`.cpp:46`) — nearest divisor of dim. **complete** indirectly (`quantizer_test.cpp:111`).
- `validate(QuantParams,dim)` (`.cpp:63`) — **complete** (`quantizer_test.cpp:304-337`).
- `code_bytes_for(QuantParams,dim)` (`.cpp:85`) — **untested** (no direct in-scope call). `train_sample_cap=100000` const (`.hpp:98`) — **untested**.

### ProductQuantizer (`ProductQuantizer.hpp:28`, `.cpp`) — PQ + OPQ (empty `rotation_`=PQ)
- `train(...)` (`.cpp:255`) — real; `opq_iters==0`→plain PQ codebook; else OPQ alternation (rotate rows, train codebook, encode, reconstruct, re-solve rotation via Procrustes, skipping re-solve after final iter so shipped codebook/rotation stay in step). **complete** (`quantizer_test.cpp:360,381,405`).
- `deserialize(istream,CodecId)` (`.cpp:203`) — real, full header/size validation + bounded alloc. **complete** (`quantizer_test.cpp:277`, `m5_quantization_persistence_test.cpp:374`).
- `encode` (`.cpp:71`), `decode` (`.cpp:88`), `make_lut` (`.cpp:118`, ADC table), `lut_distance` (`.cpp:162`), `serialize` (`.cpp:185`), accessors `codec/code_bytes=pq_dim/codebook/rotation/pq_dim/pq_bits`. All **complete** (`quantizer_test.cpp:123,195,227,246,347,388`). Confirmed: train/encode/decode/distance all real.

### ScalarQuantizer (`ScalarQuantizer.hpp:26`, `.cpp`) — per-dim int8, 4× compression
- `train` (`.cpp:26`, min/max sweep, zero-range dim→scale 1.0), `deserialize` (`.cpp:140`, bounded), `encode` (`.cpp:52`, clamps out-of-range), `decode` (`.cpp:70`), `make_lut` (`.cpp:83`), `lut_distance` (`.cpp:112`), `serialize` (`.cpp:130`). All **complete** (`quantizer_test.cpp:132,156,175,195,227,246`). Fully implemented, no stubs.

### detail/KMeans (`detail/KMeans.hpp`, `src/quant_engine/KMeans.cpp`) — namespace `elips::quant::detail`
- `kmeans` (`.cpp:182`) — real Lloyd's, deterministic strided seeding (byte-reproducible, no RNG), empty-cluster reseed, convergence break.
- `train_pq_codebook` (`.cpp:249`), `encode_pq` (`.cpp:270`), `nearest_centroid` (`.cpp:166`) — real.
- `solve_procrustes` (`.cpp:323`) — real: `M=Y^T X`, Newton-Schulz polar factor + Gram-Schmidt orthonormalize, returns identity on degenerate. No LAPACK. **complete** behaviorally (`quantizer_test.cpp:381,405`).
- `identity_matrix`/`apply_matrix`/`apply_matrix_transposed` — real.
- All of KMeans is **untested directly** but exercised transitively through PQ/OPQ training.

## vector_engine

### Metrics (`include/elips/vector_engine/Metrics.hpp`, `src/Metrics.cpp`)
Namespace `elips`. (Source lives at `src/Metrics.cpp`, not under `src/vector_engine/`.)
- `distance(Metric,span,span)` (`.cpp:104`) — ordering-normalized (smaller=closer): cosine→`1-dot`, euclidean→`sqrt(squared_l2)`, dot_product→`-dot`. **complete** (`metrics_test.cpp:13-35`; also the oracle in quantizer tests).
- `requires_normalization(Metric)` (`.cpp:117`) — true only for cosine. **complete** (`metrics_test.cpp:37`).
- Three metrics: cosine/euclidean/dot_product. Two kernels (`dot`, `squared_l2`) each with **scalar AND ARM NEON SIMD** (`dot_neon`/`sql2_neon`, `.cpp:43-69`, `#if defined(__ARM_NEON)`), runtime-dispatched once via a function-local-static `Dispatch` singleton (`.cpp:75-92`). Header comment `:15` ("SIMD is a later-phase optimization (Per.2)") is **stale** — NEON is implemented; only x86 AVX2/AVX-512 remain future ("future kernels" `.cpp:73-74`, x86 falls back to scalar).
- Also defined here but declared in `Config.hpp`: `to_string(Metric)` (`.cpp:121`), `metric_from_string()` (`.cpp:133`, throws `ConfigError`). **complete** via `config_test.cpp:71-88` (untested from `metrics_test.cpp` itself).

---

## query_engine

Namespace `elips::eql`. EQL = the Elips Query Language. **Note:** `tests/fuzz/parser_robustness_test.cpp`, despite its name, fuzzes `WAL::replay`/snapshot parsing — it provides **zero** EQL coverage.

### EQLLexer (`EQLLexer.hpp`, `src/EQLLexer.cpp`)
- `enum class TokenKind{word,number,string,punct,end}` (`.hpp:10`); `struct Token{kind;text;number;is_integer}` (`.hpp:12`).
- `tokenize(string_view)` (`.cpp:19`) — whitespace + `#` line comments; `"`-delimited strings with **no escape processing** (first `"` ends it; throws `ElipsError` on unterminated `.cpp:44`); numbers with optional leading `-`, single `.`, **no exponent/+/hex** (`std::stod`, overflow unguarded); words `[A-Za-z_][A-Za-z0-9_]*` (keywords not distinguished); punctuation, collapsing two-char `<=`,`>=`,`!=`,`==`; always appends `end`. **complete** (`eql_edge_test.cpp:26,32,38`). Note: `==` lexes as one token but the parser rejects it (only `=` is equality). Unterminated-string throw + `stod` overflow **untested**.

### AST (`AST.hpp`)
Node types: `VectorRef{literal;binding}` (`:15`), `SearchStatement{vault;query;optional top;optional threshold;Filter where;optional rank_by;projection}` (`:20`), `FetchStatement{vault;id}` (`:30`), `ScanStatement{vault;where;optional offset;optional limit}` (`:35`), `InsertStatement{vault;vector;data}` (`:42`), `DeleteStatement{vault;id}` (`:48`), `Statement = variant<Search,Fetch,Scan,Insert,Delete>` (`:53`). Plain data; **complete** (all five constructed by parser, consumed by executor, each tested).

### EQLParser (`EQLParser.hpp`, `src/EQLParser.cpp`)
- `class ParseError : public ElipsError` (`.hpp:12`); `parse(string_view)->Statement` (`.cpp:291`).
- Recursive-descent. **Real statement verbs are `seek`/`fetch`/`scan`/`place`/`erase`** (NOT `top=`/`meta`):
  - `seek in <vault> nearest <vecref>` + clauses `top <num>`/`threshold <num>`/`where <filter>`/`rank_by <ident>`(distance⇒nullopt)/`project <proj>`, terminator `yield` **required** (`.cpp:174-202`).
  - `fetch from <vault> id <string> yield` (`.cpp:216`).
  - `scan in <vault>` + `where`/`offset`/`limit`, `yield` required (`.cpp:227`).
  - `place in <vault> vector <literal>` optional `data <json-object>` — **no `yield`** (`.cpp:248`).
  - `erase from <vault> id <string>` — **no `yield`** (`.cpp:261`).
- Sub-grammars: vector literal `[n,n,…]` (empty allowed), `$ident` binding, value (string/number→int64|double/`true`|`false`→bool), filter with precedence `or < and < not/parens < comparison`, comparison leaf `in [..]`→in_set / `contains <string>`→has_substring / comparator+value→compare, comparators `=`,`!=`,`<`,`<=`,`>`,`>=` (`.cpp:160`), projection `*`→all else ident list, json object `{"key":value,…}`.
- **complete (core)** — seek/place/scan/fetch/erase, `in`+bool, malformed→ParseError, comments, offset/limit, threshold, project list, and/or/not all tested (`eql_test.cpp:11-70`, `eql_edge_test.cpp:13-103`).
- **Untested parser paths**: `rank_by` clause (both branches), EQL-level `contains`, parenthesized filter groups, comparators `in`/`!=`/`<`/`<=`/`>` via EQL (only `=`,`>=`,`in` hit), `project *`, empty `[]`, missing/unknown `$`-binding.
- **Grammar quirks**: `==` rejected (only `=`); `place`/`erase` omit the `yield` that reads require; no multi-statement input, no EOF check after a statement; `top`/`offset`/`limit` `static_cast<int>` with no validation (negatives/floats accepted).

### QueryExecutor (`QueryExecutor.hpp`, `src/QueryExecutor.cpp`)
- `execute(Statement, ElipsInstance&, bindings)->vector<SearchResult>` (`.cpp:142`, `std::visit` over an `Executor`). `ElipsInstance::query(eql, bindings)` (`.cpp:151`) = `execute(parse(eql),…)` — the public EQL entrypoint.
- `Search` (`.cpp:78`): resolves `$binding` (throws `NotFound` if absent `.cpp:21`), computes `top` (explicit, else 100000 when threshold set, else 10), `vault.seek(query,top,where,threshold)`, optional `rank_by` `stable_sort` (missing field sorts last), projection.
- `Fetch` (`.cpp:104`) hit+miss, `Scan` (`.cpp:112`, offset 0/limit SIZE_MAX), `Insert` (`.cpp:127`, returns new id), `Delete` (`.cpp:134`). **complete** (`eql_test.cpp:23,37,47`, `eql_edge_test.cpp:48,64,71,81,92`).
- **Untested**: `rank_by` sort branch (`.cpp:89-99`), unbound-`$binding` `NotFound`.

## metadata

### Filter (`metadata/Filter.hpp`, `src/Filter.cpp`)
- `enum class Comparator{eq,ne,lt,le,gt,ge}` (`.hpp:13`). `class Filter` (`.hpp:23`) — persistent immutable shared `Node` tree; `root_==nullptr`⇒match-all. `Kind{cmp,in,contains,conj,disj,neg,none}`.
- Fluent builder `field().equals/not_equals/lt/le/gt/ge()`/`one_of()`/`contains()` (AND-folded leaves, `.cpp:115-137`); static factories `compare`/`in_set`/`has_substring` (`.cpp:139-158`); combinators `and_`/`or_`(match-all operand collapses to match-all)/`not_`(not_ of match-all⇒`none`=matches nothing) (`.cpp:160-186`).
- `matches(Payload)` (`.cpp:188`) via `eval_node` (`.cpp:56`): numeric int64/double compared cross-type as double, bool/string within type, kind mismatch⇒fail; missing field⇒false; `contains` needs existing string field; `none`⇒always false.
- `exact_constraints()` (`.cpp:192`) — index-serviceable reduction: only `cmp`+`eq`, `in`, and recursive `conj` yield constraints; `contains`/`disj`/`neg`/`none`/non-eq comparator⇒`nullopt`. `matches_all()` (`.hpp:56`).
- **complete** — extensively covered (`filter_edge_test.cpp:13-127`, `transaction_filter_test.cpp:15,33,45`). **Untested branches**: `or_`/`not_` match-all collapse (`.cpp:168-182`), `exact_constraints` rejecting a non-eq `cmp` (`.cpp:204-205`).

### MetadataIndex (`metadata/MetadataIndex.hpp`, `src/MetadataIndex.cpp`)
Inverted exact-value index `map<field, map<encoded-value, set<RecordID>>>` (`.hpp:26`).
- `insert`/`remove` (`.cpp:41/47`, prunes empty buckets), `exact_candidates(Filter)` (`.cpp:67`): pulls `exact_constraints()`, `nullopt`⇒caller full-scans, unions buckets per field then **intersects across fields** short-circuiting on empty. `encode_value` (`.cpp:22`): int64→`"n:"+(double)v` at **default 6-sig-digit precision**, double→`"n:"`+17 digits, bool→`"b:0|1"`, string→`"s:"`. **complete** (`metadata_index_test.cpp:13,40,69`).
- **Latent bug (untested)**: `encode_value` casts int64 through double at 6-sig precision (`.cpp:28`) while double uses 17 (`.cpp:30`). Large ints (≥~1e7) can collide into one bucket → `exact_candidates` false positives (re-filtered by `Filter::matches`, so correctness survives but index is imprecise). `int64{42}`/`double{42.0}` intentionally both encode `"n:42"`.

---

## storage

### FileSync (`include/elips/storage/FileSync.hpp`) — header-only, `elips::detail`
- `struct SyncProbe{atomic calls;atomic fail_next;atomic fail_after}` (`:28`) + `sync_probe()` Meyers singleton (`:35`) — fault-injection/observation hook. **complete** (`m1_durability_atomicity_test.cpp:33-67,197-199`).
- `sync_file_data(int fd)` (`:42`) — honors probe injection (throws `StorageError` "(injected)"), then `_commit`(Win)/`F_FULLFSYNC`(macOS)/`fdatasync`(Linux)/`fsync`, throws on real failure. **complete** (`m1_durability_atomicity_test.cpp:42-68`).
- `sync_file_path(path)` (`:79`) — **complete** indirectly (checkpoint/reset). `sync_directory(path)` (`:110`, best-effort, no-op on Win) — **complete** indirectly. `durable_rename(tmp,final)` (`:125`) — real but **untested** directly in scope.

### Serialization (`include/elips/storage/Serialization.hpp`) — header-only, `elips::detail`
Native byte order (cross-platform normalization flagged as later hardening, `:17-18` — a design note, not a TODO).
- `put<T>`/`get<T>` (`:21`), `stream_remaining` (`:35`), `max_field_bytes=64MiB` (`:54`), `check_length(istream, needed)` (`:60`, rejects oversize **before** allocation → `StorageError`). **complete** (`m1_durability_atomicity_test.cpp:94-132`).
- `put_string`/`get_string` (`:73`), `put_payload`/`get_payload` (`:89`, u8 type-tag switch 0=int64/1=double/2=bool/3=string else `StorageError`), `put_document_attachment`/`get_document_attachment` (`:141`), `put_chunk_info`/`get_chunk_info` (`:163`), `put_embedding_lineage`/`get_embedding_lineage` (`:186`). All **complete** (round-tripped `wal_recovery_test.cpp:158-166`, bounds `m1_durability_atomicity_test.cpp:100,108`).
- `crc32c(void*,size)` (`:211`) — CRC32C/Castagnoli, static table. **complete** (tail-drop on mismatch `wal_recovery_test.cpp:91-105`, `m1_durability_atomicity_test.cpp:111-132`).

### WAL (`include/elips/storage/WAL.hpp`, `src/WAL.cpp`)
`class WAL` non-copyable/non-movable (all four deleted `.hpp:52-55`).
- `enum class Op:uint8_t{insert=1,erase=3,insert_ex=4,txn_begin=5,txn_commit=6,insert_q=7}` (`.hpp:23`, value 2 skipped; `insert_ex`=insert+doc/chunk/lineage, `insert_q`=quantized carrying `codes`). `struct Entry{op;vault;id;vector;payload;optional doc/chunk/lineage;codes;codec}` (`.hpp:36`). **complete** (`wal_recovery_test.cpp:118-166`, `m1_durability_atomicity_test.cpp:213-217`).
- ctor (`.cpp:72`, O_WRONLY|O_CREAT|O_APPEND, throws `StorageError` on fail), dtor (`.cpp:84`). **complete**.
- `append_insert` (`.cpp:131`), `append_erase` (`.cpp:151`), `append_txn_begin`/`append_txn_commit` (`.cpp:156`), `sync` (`.cpp:125`) — **complete** (`m1_durability_atomicity_test.cpp:47,66,115,142,213-221`, `wal_recovery_test.cpp:113-122`).
- `append_insert_quantized` (`.cpp:141`, Op::insert_q) — real but **untested** in scope. `reset()` (`.cpp:166`, truncate + durable sync) — real but **untested** directly in scope.
- `replay(path)` (`.cpp:224`) — **the central tested surface**. Reads whole file into a blob, parses record-by-record over a zero-copy `SpanBuf` streambuf (makes replay O(n), `.cpp:191-220`; linearity asserted `m1_durability_atomicity_test.cpp:136-161`). Per record: validate magic `0xE1105E01` (clean stop otherwise), decode op/vault/id, `check_length`-bounded field reads, malformed→corrupt-tail stop, trailing CRC32C mismatch→stop. Transaction framing pass (`.cpp:315-343`): entries inside an unterminated `txn_begin` window are dropped. **complete** (`wal_recovery_test.cpp:91-105`, `m1_durability_atomicity_test.cpp:111-132,206-222`; plus `tests/fuzz/fuzz_wal_replay.cpp` separate target).
- Private `append` (`.cpp:112`, one atomic-tail `write_all` per record + optional fsync), `write_all` (`.cpp:94`, partial-write/EINTR loop), file-local `encode_body` (`.cpp:28`, promotes insert→insert_ex when extras present, never promotes insert_q). **complete**.

## kernel

### LockManager (`include/elips/kernel/LockManager.hpp`, `src/LockManager.cpp`) — cross-process advisory lock
- `class LockConflict : public ElipsError` (`.hpp:11`) — **complete** (`single_writer_test.cpp:19-20`, `multi_reader_test.cpp:50,123`).
- `enum class LockMode{exclusive,shared}` (`.hpp:16`) — exclusive **complete**, shared **untested** (no in-scope caller passes `shared`).
- `class LockManager` RAII (`.hpp:21`, non-copyable, move-ctor only): ctor (`.cpp:18`, POSIX `flock(LOCK_EX|LOCK_SH|LOCK_NB)` / Windows `LockFileEx(LOCKFILE_FAIL_IMMEDIATELY)`, throws `LockConflict` on contention), dtor (`.cpp:60`, unlock+close). **complete** (`single_writer_test.cpp:18-29`). Move-ctor (`.cpp:74`) **untested** directly.

## Config

### Config (`include/elips/Config.hpp`)
- `enum class Metric{cosine,euclidean,dot_product}` (`:27`) — **complete** (`config_test.cpp:71-88`). `IndexType{graph,exact}` (`:31`) — **complete** (`config_test.cpp:62`). `Durability{paranoid,standard,relaxed,ephemeral}` (`:38`, paranoid+standard both flush per record today) — **complete** (`config_test.cpp:53`, `durability_test.cpp:27-69`). `AccessMode{read_write,read_only}` (`:40`) — **untested**.
- `to_string(Metric)`/`metric_from_string()` (`:42`, defined in `src/Metrics.cpp`) — **complete** (`config_test.cpp:71-88`).
- `open(path, Config)` (`:23`, body out of scope in `Database.cpp`) — **complete** (nearly every integration test).
- `struct GraphParams{max_connections=16(M);ef_construction=200;ef_search=50;compaction_ratio=0.2}` (`:46`) — first three **complete** (`config_test.cpp:35-51,90-100`), `compaction_ratio` **untested**.
- `class Config` fluent builder (`:56`), copyable. **Tested**: `dimension`/`metric`/`index`/`graph_params`/`durability` (`config_test.cpp:21-108`). **Untested (real bodies)**: `quantization`/`has_quantization`, `access_mode`, `segmented_storage` (default true), `metadata_acceleration` (default true), the whole `text_embedder`/`local_text_embedder`/`auto_text_embedder`/`text_embedder_info` cluster (`:98-163`), `attach_runtime_text_embedder`/`remember_expected_text_embedder`, `attach_quantizer`/`quantizer`/`has_trained_quantizer` (held on Config so rebuild/vacuum/compact don't drop it, comment `:182-186`), and `gpu`/`has_gpu` (behind `ELIPS_GPU_ENABLED`).

---

## facade (elips.hpp + Database.cpp)

`src/Database.cpp` (1983 lines) holds the whole facade **except** `ElipsInstance::query` (EQL), which lives in `src/QueryExecutor.cpp:151`. No TODO/FIXME/throwing-stub; all throws are legitimate validation.

### Real API verbs (`include/elips/elips.hpp`)
| Verb | Decl / impl | Meaning |
|------|-------------|---------|
| `place` | `elips.hpp:130` / `Database.cpp:869` | insert a raw `Vector` (+payload/doc/chunk/lineage) |
| `place_document` | `:135` / `1005` | insert from **text** (embeds it) |
| `place_many` | `:140` / `1050` | batch insert of `Record`s |
| `place_record` | `:147` / `953` | load-path verbatim store (snapshot/segment/WAL replay) |
| `seek` | `:149` / `1178` | vector KNN |
| `seek_text` | `:152` / `1257` | embed text then KNN |
| `seek_hybrid` | `:155` / `1272` | vector KNN + lexical overlap re-rank |
| `explain_seek` | `:160` / `1316` | returns a `QueryPlan`, runs no search |
| `scan` | `:165` / `1324` | filtered iteration offset/limit |
| `fetch` | `:169` / `1346` | by-id lookup |
| `erase` | `:170` / `1355` | delete by id |
| `info` | `:172` / `1564` | `VaultInfo` summary |
| `records` | `:178` / `864` | copy of the record map |
| `rebuild_index` | `:183` / `1409` | drop & re-insert into fresh index |
| `vacuum`/`pending_removals` | `:188/:190` / `1553/1559` | tombstone reclaim/count |
| `quantize` | `:208` / `1470` | train codebook + compress in place |
| `query` (EQL) | `:306` / **`QueryExecutor.cpp:151`** | parse+execute EQL |
| `checkpoint`/`compact`/`close`/`abandon` | `:310-322` / `1662/1731/1754/1597` | lifecycle |
| `begin_transaction` | `:304` / `1875` | returns a `Transaction` |

**Vocabulary reality check vs the task's speculative list**: there is **no** `ingest`, `probe`, `meta`, `snapshot`, or `arena` verb. Insert verbs are `place*`; search verbs are `seek*`; the container is `Vault` (via `ElipsInstance::vault(name)` `elips.hpp:301`); `commit` is only `Transaction::commit` (`elips.hpp:405`/`Database.cpp:1913`); "snapshot" is an on-disk file format (internal anon-namespace functions), not a public verb; `top` is a `seek` parameter, not a verb.

### Free function
- `elips::open(path, Config={})` (`elips.hpp:370`/`Database.cpp:1775`) — **complete**. `:memory:` vs on-disk, IDENTITY read/write + dimension reconciliation, text-embedder resolution/rehydration, GPU config, snapshot-or-segment load, WAL replay (`Database.cpp:1842-1858`), read-only locking. Tested `roundtrip_test.cpp:56,71`.

### Public types
- `struct VaultInfo{count;dimension;metric;codec;code_bytes}` (`elips.hpp:87`) — **complete** (`roundtrip_test.cpp:86`, `m5_quantization_persistence_test.cpp`).
- `enum class QueryStrategy{ann_index,exact_candidates,full_scan,text_probe,hybrid_fusion}` (`:98`) — **caveat**: `seek_locked` (`Database.cpp:1195-1211`) only branches on `exact_candidates`/`full_scan`→else ANN; `text_probe` is only produced by `plan_seek` when records empty/top==0; `hybrid_fusion` is a plan **label** only (`seek_hybrid` never reads `QueryStrategy`). The enum is wider than the dispatch. Labels tested via `explain_seek` (`roundtrip_test.cpp:203-222`).
- `struct QueryPlan` (`:106`) — **complete** (`roundtrip_test.cpp:211-221`).

### class Vault (`elips.hpp:122`)
Named partition owning an `IndexPort`, `map<RecordID,Record>`, `MetadataIndex`, own `shared_mutex`. All verbs real (`_locked` helpers under lock):
- `place` (`Database.cpp:869`) — normalizes if metric requires, WAL-logs, replaces by id, inserts into index+metadata, and if a codebook is attached **encodes and drops the fp32 vector** (`:920-924`). **complete** (`roundtrip_test.cpp:59,134`).
- `place_document` (`:1005`) — throws `ConfigError` (recovery guidance) if no embedder; auto-fills ChunkInfo/EmbeddingLineage; embeds outside lock. **complete** (`roundtrip_test.cpp:165`, `local_text_embedder_test.cpp:126`).
- `place_record` (`:953`) — load path, coded record stored byte-for-byte (validates `codes.size()==code_bytes()`), decoded once for index. **complete** (reopen/WAL replay paths, `m5_quantization_persistence_test.cpp`).
- `place_many` (`:1050`) — real, checks writability up-front. **untested (C++)** — only `tests/python/test_bindings.py:362`.
- `seek` (`:1178`) — plans, then exact-candidate/full-scan/ANN with **widening beam** (re-probes `fetch*=4` until filter satisfied or exhausted, `:1228-1254`). **complete** (`roundtrip_test.cpp:62,93,150`, `transaction_filter_test.cpp:26`).
- `seek_text` (`:1257`) — **complete** (`roundtrip_test.cpp:196`, `local_text_embedder_test.cpp:145,237`).
- `seek_hybrid` (`:1272`) — real; over-fetches `top*5`, blends `(1-w)*distance + w*(1-lexical_overlap)` (Jaccard-ish `lexical_overlap_score` `:117`). **untested (C++)** — only `tests/python/test_bindings.py:305`.
- `explain_seek` (`:1316`), `scan` (`:1324`, hydrates/decodes), `fetch` (`:1346`), `erase` (`:1355`), `info` (`:1564`), `records` (`:864`, returns a copy for thread-safety), `rebuild_index` (`:1409`), `vacuum`/`pending_removals` (`:1553/1559`). All **complete** (`roundtrip_test.cpp:88,125,211`, `transaction_filter_test.cpp:33`, `scan_test.cpp`, `m3_index_health_test.cpp:253-257`).
- `quantize` (`:1470`) — throws `ConfigError` if no codec / already quantized / empty vault; trains on bounded strided sample (`quant::train_sample_cap`), encodes all, drops fp32, rebuilds index. **complete** (`m5_quantization_persistence_test.cpp:452,459,469` throw cases; `:141,176,379,455,482,509` success).
- `install_quantizer` (`:1448`), `seal`/`sealed`/`set_read_only`/`read_only` + `ensure_writable` (throws `StorageError`), `restore_for_undo` (`:1378`, lossless coded restore). **complete** (`roundtrip_test.cpp:256` read-only throw).

### class ElipsInstance (`elips.hpp:290`)
Owns vaults + WAL + lock (+ optional GPU); `recursive_mutex` registry.
- `vault`/`vault_locked` (`:1602/1607`, lazy create), `list_vaults` (`:1651`), `begin_transaction` (`:1875`), `query` (EQL, `QueryExecutor.cpp:151`), `checkpoint`/`checkpoint_locked` (`:1662/1667`, snapshot or segmented manifest+segments, durable-rename, truncate WAL), `compact` (`:1731`), `quantize(name)`/`quantize_all` (`:1529/1547`, persistent DB checkpoints after), `vacuum` (`:1743`), `close` (`:1754`), `abandon` (`:1597`, no checkpoint = crash sim), destructor (`:1581`, best-effort checkpoint). All **complete** (`durability_test.cpp:43,94`, `roundtrip_test.cpp:78,156,235,244`, `m1/m3/m5` tests, `m2_thread_safety_test.cpp:147`).
- `adopt_vault`/`attach_wal` (`:1631/1642`) — real plumbing for `open`.

### class Transaction / TransactionVault (`elips.hpp:375/389`)
- `TransactionVault::place`/`erase` (`:1966/1973`) enqueue only (`place` pre-generates id); `enqueue_place` validates dim+finiteness eagerly (`:1883`). **complete** (`transaction_filter_test.cpp:75`).
- `commit` (`:1913`) — **atomic**: instance lock, pre-check every vault for read-only/sealed, WAL `txn_begin`, apply building undo log, on exception `undo()` reverse-order + rethrow **without** commit marker (replay discards batch). **complete** (`transaction_filter_test.cpp:53`, `m1_durability_atomicity_test.cpp`).
- `rollback` + destructor auto-rollback (`:1877`). **complete** (`transaction_filter_test.cpp:65`).

### Internal persistence (anon namespace in Database.cpp)
On-disk format v3 (`snapshot_version` `:57`). Real, codec-aware, length-bounded: `put/get_record` (`:432/460`), `put/get_codebook` (`:516/529`), `write/read_identity` (`:548/564`), `write_snapshot_file`/`load_snapshot_file` (`:577/682`), segment variants (`:611/719`), text-embedder manifest r/w (`:205/234`), `resolve_text_embedder_for_open` (`:285`). Exercised by reopen/segmented/read-only tests (`roundtrip_test.cpp:224`, `m5_*`).

---

## text_engine

### TextEmbedderPort (`include/elips/text_engine/TextEmbedderPort.hpp`)
- `enum class TextEmbedderKind{external,local_builtin}` (`:14`); `struct TextEmbedderInfo{kind;provider;model;revision;backend;dimension;fingerprint;storage_path;rehydratable;loaded;auto_attached}` (`:16`).
- `class TextEmbedderPort` (`:30`) — abstract; pure virtual `embed` (`:34`); `embed_batch` has a **real default** (loops `embed`, `:36`); `info()` assembles a `TextEmbedderInfo` (`:67`); the rest are sensible default hooks, not stubs. **complete** via toy subclasses in `roundtrip_test.cpp:19`, `local_text_embedder_test.cpp:20`.

### LocalTextEmbedder (`LocalTextEmbedder.hpp`, `src/LocalTextEmbedder.cpp`)
- `struct LocalTextEmbedderOptions{model="default";revision="v1";storage_path;dimension}` (`.hpp:12`).
- `class LocalTextEmbedder final : public TextEmbedderPort` (`.hpp:19`, impl `.cpp:268`). **What it REALLY is: a deterministic feature-hashing embedder — NOT a neural model and NOT a tokenizer-based ML embedder.** It is a genuine working embedder of its own kind, not a no-op placeholder:
  1. `tokenize_terms` (`.cpp:196`) lowercases + splits on non-alphanumerics.
  2. `collect_features` (`.cpp:215`) emits tokens, optional adjacent bigrams, bounded char n-grams.
  3. `embed` (`.cpp:344`) — `fnv1a(feature)^seed`, then `projections_per_feature` rounds of splitmix64-style `mix64`, each hashed to a dim index with a sign bit, accumulating `sign/(projection+1)` → **signed random-projection / hashing-trick** into a fixed-dim vector. Deterministic (`local_text_embedder_test.cpp:107`).
  - Two "models" are fixed hyperparameter sets (`resolve_model` `.cpp:76`): `default@v1`, `compact@v1`; unknown→`ConfigError`.
  - "Model loading" is real but trivial: `ensure_artifact_written` (`.cpp:156`) writes a tiny `.localembed` binary (magic `0xE1105E21`) of hyperparameters+fingerprint; `ensure_loaded` (`.cpp:281`) lazily reads/validates it (throws `StorageError` on mismatch/truncation). `provider="elips-local"`, `backend="builtin-hash"`.
  - **complete** (`local_text_embedder_test.cpp:79,98-105,110,150`).
- Free fns `describe_local_text_embedder` (`.cpp:245`), `make_local_text_embedder`, `make_default_local_text_embedder` (`.cpp:415/419`). **complete** (`local_text_embedder_test.cpp:88`).

---

## cli (`cli/src/main.cpp`, `cli/src/Json.hpp`)

Binary `elips <command> <db_path> [--key value]…` (usage `main.cpp:296`, dispatch `:317-328`). 10 real commands: `info` (`:104`), `vaults` (`:133`), `query` (`:141`, EQL from `--eql`/`--file`→JSON lines), `checkpoint` (`:159`), `export` (`:166`, scan `--vault`→JSON lines, warns on compressed), `import` (`:219`), `verify` (`:238`, OK/CORRUPT + exit 1), `stats` (`:254`), `bench` (`:263`), `quantize` (`:190`). Config via `config_from_args` (`:48`); `open_db` (`:82`) reconciles IDENTITY; top-level try/catch→`error:`+exit 1 (`:330`). **Real, but untested at the binary level** — no C++ test drives `main()`; the underlying library logic is tested through the API.

`Json.hpp` — `namespace elips::cli::json`, a minimal hand-written JSON reader/writer scoped to CLI import/export, explicitly "not a general JSON library" (`:14`). `escape` (`:18`, only `"`/`\`, no control/unicode), `dump_value`/`dump_record` (`:30/47`), `class Reader` (`:67`), `parse_record` (`:189`). Real; **untested** (no dedicated test includes it).

## benchmarks (`benchmarks/BenchMain.cpp`)

Standalone exe `elips_bench [--count N] [--dim D] [--queries Q]` (`:137`). Real performance harness (no assertions, not a gtest): `bench_insert_and_search` (`:45`, builds `HierarchicalGraphIndex`+`ExactIndex` **directly, not via the facade**, measures insert rec/s, p50/p95/p99 latency, recall@10 vs exact), `bench_recovery` (`:103`, insert→`abandon()`→reopen→time WAL replay). The only in-scope file driving the raw index classes directly.

---

## Headline findings (biggest stubs / mismatches)

1. **Nothing is a stub.** Zero `TODO`/`FIXME`/"not implemented"/empty-body across the entire in-scope core. Every declared method has a real implementation. The core is far more built-out than the task's framing suggested.
2. **Only one deliberately-unimplemented feature**: `QuantParams::refine_fraction` (exact reranking of quantized candidates), `QuantParams.hpp:46`. It is unreferenced by any code **and the ADR it cites — ADR-0011 — does not exist** (docs/adr stops at 0010). This is the single real "to-build" item.
3. **The task's vocabulary largely does not match the code.** No `ingest`, `probe` (as a verb), `meta`, `arena`, `top=`, or `IndexPort`-as-"port-named-that" verbs exist. Real verbs: insert=`place`/`place_document`/`place_many`/`place_record`, query=`seek`/`seek_text`/`seek_hybrid`, delete=`erase`, iterate=`scan`, point-get=`fetch`. Container=`Vault`. The port interface is `IndexPort` (type, not verb). EQL statement keywords are `seek`/`fetch`/`scan`/`place`/`erase` with `nearest/top/threshold/where/project/rank_by/yield`.
4. **`HierarchicalGraphIndex` is a real, complete HNSW** (probabilistic levels, beam search, Algorithm-4 diversity neighbor selection, soft-delete tombstones + adaptive-ef + auto-compaction), tested to recall@10 ≥ 0.90 vs the exact oracle. Not a placeholder.
5. **PQ/OPQ/SQ8 quantization is fully implemented** (train/encode/decode/ADC-LUT/serialize, OPQ via from-scratch Newton-Schulz Procrustes, no LAPACK) and well-tested at the component level.
6. **Biggest coverage gap: the entire quantized/compressed read path through the indexes and snapshots is untested in C++.** `ExactIndex`/`HierarchicalGraphIndex` `export_snapshot`/`import_snapshot`, their `compressed()`-true branches, `IndexSnapshot`/`IvfSnapshot`/`PqSnapshot`, `make_index`/`IndexFactory`, `WAL::append_insert_quantized`, and `WAL::reset`/`durable_rename` all have real bodies but no in-scope test. (Vault-level quantization *is* tested via `m5_*`.)
7. **`QueryStrategy` enum is wider than the dispatch.** `text_probe` and `hybrid_fusion` are plan **labels** only — `seek_locked` never dispatches on them and `seek_hybrid` ignores the enum entirely. Dead flexibility, not a bug.
8. **`Vault::place_many` and `Vault::seek_hybrid` have real bodies but zero C++ test coverage** (only Python binding tests). `seek_hybrid`'s lexical-overlap fusion is otherwise unverified in-tree.
9. **Latent precision bug in `MetadataIndex::encode_value`** (`MetadataIndex.cpp:28`): int64 keys are stringified through `(double)` at default 6-sig-digit precision while doubles use 17, so distinct large ints (≥~1e7) can collide into one bucket → false-positive candidates. Correctness is saved only because `Filter::matches` re-filters; untested.
10. **Several stale comments / dangling refs**: `IndexPort.hpp:14-15` says HNSW "plugs in later" (it already ships); `HierarchicalGraphIndex` comments reference a non-existent `unlink()` method; `Metrics.hpp:15` calls SIMD "a later-phase optimization" though NEON kernels are implemented (only x86 AVX remains). The `std::expected` polyfill (`Expected.hpp`) is dead on this C++23 toolchain, and EQL `==` lexes but is rejected by the parser.

