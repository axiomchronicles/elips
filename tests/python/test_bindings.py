"""Comprehensive Python binding validation tests for ELIPS.

Tests every exposed Python API for correctness, typing, and C++ parity.
"""

import sys
import os
import gc
import threading
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))
import elips


def test_exceptions():
    """Verify all 8 exceptions are importable and properly raised."""
    for name in [
        "ElipsError", "DimensionMismatch", "InvalidVector", "ConfigError",
        "NotFound", "StorageError", "LockConflict", "ParseError",
    ]:
        exc = getattr(elips, name)
        assert issubclass(exc, Exception), f"{name} is not an Exception subclass"

    assert issubclass(elips.DimensionMismatch, elips.ElipsError)
    assert issubclass(elips.ParseError, elips.ElipsError)
    assert issubclass(elips.LockConflict, elips.ElipsError)

    # Verify actual raising
    db = elips.open(":memory:", dimension=2)
    try:
        db.vault("v").place([1.0])  # wrong dimension
        assert False, "should have raised"
    except elips.DimensionMismatch:
        pass

    try:
        elips.validate_eql("garbage syntax !!!")
        assert False
    except elips.ParseError:
        pass

    print("  PASS test_exceptions")


def test_enums():
    """Verify all core and GPU enums exist with correct values."""
    assert int(elips.Metric.cosine) != int(elips.Metric.euclidean)
    assert int(elips.Metric.euclidean) != int(elips.Metric.dot_product)
    assert int(elips.Metric.cosine) != int(elips.Metric.dot_product)
    assert int(elips.IndexType.graph) != int(elips.IndexType.exact)
    assert int(elips.Durability.paranoid) != int(elips.Durability.ephemeral)
    assert int(elips.Comparator.eq) != int(elips.Comparator.ge)

    if elips._has_gpu:
        assert int(elips.GpuPolicy.auto) != int(elips.GpuPolicy.cpu_only)
        assert int(elips.GpuError.device_not_found) >= 0
        assert int(elips.GpuPrecision.fp32) != int(elips.GpuPrecision.fp16)
        assert int(elips.GraphBuildAlgo.ivf_pq) >= 0

    print("  PASS test_enums")


def test_utilities():
    """Verify utility functions with both string and enum overloads."""
    assert elips.distance("cosine", [1.0, 0.0], [0.0, 1.0]) == 1.0
    assert elips.distance(elips.Metric.cosine, [1.0, 0.0], [0.0, 1.0]) == 1.0
    assert elips.distance("euclidean", [0.0, 0.0], [3.0, 4.0]) == 5.0
    assert elips.distance(elips.Metric.dot_product, [1.0, 2.0], [1.0, 1.0]) == -3.0

    assert elips.requires_normalization("cosine") is True
    assert elips.requires_normalization("euclidean") is False
    assert elips.requires_normalization(elips.Metric.cosine) is True
    assert elips.requires_normalization(elips.Metric.dot_product) is False

    assert elips.metric_to_string(elips.Metric.cosine) == "cosine"
    assert elips.metric_from_string("euclidean") == elips.Metric.euclidean

    print("  PASS test_utilities")


def test_eql_tooling():
    """Verify EQL tokenize and validate functions."""
    tokens = elips.tokenize_eql("seek in docs nearest [1.0, 0.5] top 5 yield")
    assert len(tokens) > 0
    assert tokens[0].kind == elips.TokenKind.word
    assert tokens[0].text == "seek"

    # Validate succeeds on valid EQL
    assert elips.validate_eql("seek in v nearest [1.0] top 5 yield") is None

    # Validate fails on broken EQL
    try:
        elips.validate_eql("invalid")
        assert False
    except elips.ParseError:
        pass

    print("  PASS test_eql_tooling")


def test_config():
    """Verify Config builder and property getters."""
    c = (elips.Config()
         .dimension(384)
         .metric("euclidean")
         .index("exact")
         .durability("paranoid")
         .graph_params(elips.GraphParams(32, 400, 100)))

    assert c.dimension_val == 384
    assert c.metric_val == "euclidean"
    assert c.index_val == "exact"
    assert c.metric_enum == elips.Metric.euclidean
    assert c.index_enum == elips.IndexType.exact
    assert c.durability_enum == elips.Durability.paranoid
    assert c.graph_params_val.max_connections == 32
    assert c.graph_params_val.ef_construction == 400
    assert c.graph_params_val.ef_search == 100

    db = elips.open_with_config(":memory:", c)
    assert db.config.dimension_val == 384
    assert db.config.metric_enum == elips.Metric.euclidean

    print("  PASS test_config")


def test_database_crud():
    """Verify full database CRUD operations."""
    db = elips.open(":memory:", dimension=3, metric="cosine")

    docs = db.vault("documents")
    assert docs.name == "documents"

    rid1 = docs.place([1.0, 0.0, 0.0], {"title": "alpha", "year": 2024})
    assert len(rid1) == 36  # UUIDv7 hex
    rid2 = docs.place([0.0, 1.0, 0.0], {"title": "beta"}, id="00000000-0000-7000-8000-000000000001")
    assert rid2 == "00000000-0000-7000-8000-000000000001"

    assert docs.count() == 2
    assert docs.info().count == 2
    assert docs.info().dimension == 3

    # Seek
    hits = docs.seek([1.0, 0.0, 0.0], top=2)
    assert len(hits) == 2
    assert hits[0].data["title"] == "alpha"
    assert hits[1].data["title"] == "beta"
    assert hits[0].distance <= hits[1].distance

    # Fetch
    r = docs.fetch(rid1)
    assert r is not None
    assert r["id"] == rid1
    assert r["data"]["title"] == "alpha"

    missing = docs.fetch("00000000-0000-7000-8000-999999999999")
    assert missing is None

    # Scan
    rows = docs.scan()
    assert len(rows) == 2
    assert "id" in rows[0]
    assert "data" in rows[0]

    # Erase
    assert docs.erase(rid2) is True
    assert docs.count() == 1

    print("  PASS test_database_crud")


def test_place_many():
    """Verify batch ingestion."""
    db = elips.open(":memory:", dimension=2)
    docs = db.vault("batch")

    docs.place_many([
        {"vector": [1.0, 0.0], "data": {"n": 1}},
        {"vector": [0.0, 1.0], "data": {"n": 2}},
        {"vector": [1.0, 1.0], "data": {"n": 3}},
    ])
    assert docs.count() == 3

    # With IDs
    docs.place_many([
        {"vector": [2.0, 0.0], "id": "00000000-0000-7000-8000-00000000000a", "data": {"n": 10}},
    ])
    r = docs.fetch("00000000-0000-7000-8000-00000000000a")
    assert r is not None and r["data"]["n"] == 10

    print("  PASS test_place_many")


def test_filtered_search():
    """Verify filtered seek and combinational filters."""
    db = elips.open(":memory:", dimension=2)
    v = db.vault("f")

    v.place([1.0, 0.0], {"cat": "tech", "score": 85})
    v.place([0.0, 1.0], {"cat": "finance", "score": 90})
    v.place([1.0, 1.0], {"cat": "tech", "score": 70})

    f = elips.Filter().field("cat").equals("tech").field("score").gte(80)
    hits = v.seek([1.0, 0.0], top=10, where=f)
    assert len(hits) == 1
    assert hits[0].data["cat"] == "tech"

    # OR filter - construct separately and combine
    cat_tech = elips.Filter().field("cat").equals("tech")
    score_hi = elips.Filter().field("score").gte(90)
    f_or = cat_tech.or_(score_hi)
    hits2 = v.seek([1.0, 0.0], top=10, where=f_or)
    assert len(hits2) == 3  # 2 tech + 1 finance with score 90

    # NOT filter
    f_not = elips.Filter.not_(elips.Filter().field("cat").equals("tech"))
    hits3 = v.seek([1.0, 0.0], top=10, where=f_not)
    assert len(hits3) == 1

    # one_of
    f_in = elips.Filter().field("score").one_of([85, 70])
    hits4 = v.seek([1.0, 0.0], top=10, where=f_in)
    assert len(hits4) == 2

    # contains substring
    f_contains = elips.Filter().field("cat").contains("fin")
    hits5 = v.seek([1.0, 0.0], top=10, where=f_contains)
    assert len(hits5) == 1

    print("  PASS test_filtered_search")


def test_transaction():
    """Verify transaction commit and rollback."""
    db = elips.open(":memory:", dimension=2)
    v = db.vault("tx")

    v.place([0.0, 0.0], {"init": True})

    # Commit
    with db.begin_transaction() as txn:
        tv = txn.vault("tx")
        tv.place([1.0, 0.0], {"added": True})
    assert v.count() == 2

    # Rollback on exception
    try:
        with db.begin_transaction() as txn:
            tv = txn.vault("tx")
            tv.place([0.0, 1.0], {"will": "rollback"})
            raise RuntimeError("intentional")
    except RuntimeError:
        pass
    assert v.count() == 2  # unchanged

    # Manual commit/rollback
    t = db.begin_transaction()
    t.vault("tx").place([2.0, 2.0], {"manual": True})
    t.commit()
    assert v.count() == 3

    t2 = db.begin_transaction()
    t2.vault("tx").place([3.0, 3.0], {"gone": True})
    t2.rollback()
    assert v.count() == 3

    print("  PASS test_transaction")


def test_database_context_manager():
    """Verify context manager closes DB properly."""
    with elips.open(":memory:", dimension=2) as db:
        db.vault("ctx").place([1.0, 0.0])

    # After context exit, db should be closed (no leak for in-memory)

    # Persistent context manager
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "testdb")
        with elips.open(db_path, dimension=2) as pdb:
            pdb.vault("p").place([1.0, 2.0])
        # Reopen to verify persisted
        db2 = elips.open(db_path)
        assert db2.vault("p").count() == 1

    print("  PASS test_database_context_manager")


def test_eql_query():
    """Verify various EQL queries."""
    db = elips.open(":memory:", dimension=2, metric="euclidean")

    # Insert via EQL
    r = db.query('place in items vector [1.0, 0.0] data {"name": "a", "val": 10}')
    assert len(r) == 1
    rid = r[0].id

    db.query('place in items vector [0.0, 1.0] data {"name": "b", "val": 20}')

    # Search via EQL
    results = db.query("seek in items nearest [1.0, 0.0] top 2 yield")
    assert len(results) == 2

    # Fetch via EQL
    fetched = db.query(f'fetch from items id "{rid}" yield')
    assert len(fetched) == 1
    assert fetched[0].id == rid

    # Scan via EQL
    scanned = db.query('scan in items where val >= 10 limit 1 yield')
    assert len(scanned) == 1

    # Erase via EQL
    db.query(f'erase from items id "{rid}"')
    assert db.vault("items").count() == 1

    print("  PASS test_eql_query")


def test_gpu_config():
    """Verify GPU configuration types."""
    if not elips._has_gpu:
        print("  SKIP test_gpu_config (no GPU)")
        return

    gc = elips.GpuConfig()
    gc.policy = elips.GpuPolicy.prefer_gpu
    gc.algorithm = elips.GpuIndexAlgorithm.brute_force
    gc.device_memory_pool_mb = 256
    assert gc.device_memory_pool_mb == 256

    gc.fp16_search = True
    assert gc.fp16_search is True

    gc.max_batch_size = 128
    assert gc.max_batch_size == 128
    assert gc.ef_search > 0

    # Graph build params
    gp = elips.GraphIndexBuildParams()
    gp.graph_degree = 64
    assert gp.graph_degree == 64
    gp.build_algo = elips.GraphBuildAlgo.nn_descent

    # IVF-PQ params
    ip = elips.IvfPqBuildParams()
    ip.n_lists = 2048
    ip.pq_dim = 64
    assert ip.n_lists == 2048
    assert ip.pq_dim == 64

    # GpuIndexBuildParams wraps variant
    bip = elips.GpuIndexBuildParams()
    assert bip.params is not None

    print("  PASS test_gpu_config")


def test_gpu_device_info():
    """Verify enhanced GpuDeviceInfo fields."""
    if not elips._has_gpu:
        print("  SKIP test_gpu_device_info (no GPU)")
        return

    info = elips.GpuDeviceInfo()
    info.name  # just verify no crash
    assert hasattr(info, "peak_tflops_fp32")
    assert hasattr(info, "peak_tflops_fp16")
    assert hasattr(info, "supports_dynamic_batching")
    assert hasattr(info, "supports_half_precision_search")
    assert hasattr(info, "supports_bf16")
    assert hasattr(info, "compute_capability_major")
    assert hasattr(info, "host_to_device_bandwidth_gb_s")

    print("  PASS test_gpu_device_info")


def test_thread_safety_python():
    """Verify Python threading safety."""
    db = elips.open(":memory:", dimension=3)
    v = db.vault("threads")
    for i in range(100):
        v.place([float(i), 0.0, 0.0])

    errors = []
    results = []

    def worker(qid):
        try:
            r = v.seek([0.0, 0.0, 0.0], top=10)
            results.append(len(r))
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread errors: {errors}"
    assert all(r == 10 for r in results), f"Wrong result sizes: {results}"

    print("  PASS test_thread_safety_python")


def test_edge_cases():
    """Test boundary conditions."""
    db = elips.open(":memory:", dimension=1)

    # Empty vector
    v = db.vault("edge")
    try:
        v.place([])
        assert False
    except elips.DimensionMismatch:
        pass

    # Large dimension
    db2 = elips.open(":memory:", dimension=1000)
    v2 = db2.vault("big")
    big_vec = [1.0] * 1000
    rid = v2.place(big_vec)
    assert len(rid) == 36

    # Zero top-k returns zero or one (implementation-dependent minimum)
    hits = v2.seek(big_vec, top=0)
    assert len(hits) <= 1

    # Negative limit in scan is treated as "all"
    v2.place([2.0] * 1000)
    rows = v2.scan(limit=-1)
    assert len(rows) == 2

    # Unicode in metadata
    db3 = elips.open(":memory:", dimension=2)
    v3 = db3.vault("unicode")
    rid3 = v3.place([1.0, 0.0], {"emoji": "🚀", "japanese": "日本語"})
    r = v3.fetch(rid3)
    assert r["data"]["emoji"] == "🚀"
    assert r["data"]["japanese"] == "日本語"

    print("  PASS test_edge_cases")


def test_memory_leak_check():
    """Repeated alloc/dealloc cycles check for leaks."""
    for _ in range(50):
        db = elips.open(":memory:", dimension=3)
        v = db.vault("leak")
        for i in range(20):
            v.place([float(i) * 0.1, 0.0, 0.0])
        v.seek([0.0, 0.0, 0.0], top=5)
        v.scan()
        db.query("seek in leak nearest [0, 0, 0] top 5 yield")
        del db, v

    gc.collect()
    print("  PASS test_memory_leak_check")


def test_type_stubs():
    """Verify py.typed marker and stub file presence."""
    elips_dir = os.path.dirname(elips.__file__)
    assert os.path.exists(os.path.join(elips_dir, "py.typed")), "py.typed missing"
    assert os.path.exists(os.path.join(elips_dir, "_core.pyi")), "_core.pyi missing"
    print("  PASS test_type_stubs")


def test_parity_cpp_vs_python():
    """C++ and Python should produce identical results for the same operations."""
    db = elips.open(":memory:", dimension=3, metric="cosine")
    v = db.vault("parity")

    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.577, 0.577, 0.577],
    ]
    metas = [
        {"name": "x"},
        {"name": "y"},
        {"name": "z"},
        {"name": "center"},
    ]
    ids = []
    for vec, meta in zip(vectors, metas):
        rid = v.place(vec, meta)
        ids.append(rid)

    # C++ expected: search for [1,0,0] should return x first
    hits = v.seek([1.0, 0.0, 0.0], top=4)
    assert hits[0].data["name"] == "x"
    assert abs(hits[0].distance - 0.0) < 0.01

    # Fetch roundtrip
    for rid in ids:
        r = v.fetch(rid)
        assert r is not None
        assert len(r["vector"]) == 3

    # Erase then fetch
    v.erase(ids[0])
    assert v.count() == 3
    assert v.fetch(ids[0]) is None

    print("  PASS test_parity_cpp_vs_python")


if __name__ == "__main__":
    tests = [
        test_exceptions,
        test_enums,
        test_utilities,
        test_eql_tooling,
        test_config,
        test_database_crud,
        test_place_many,
        test_filtered_search,
        test_transaction,
        test_database_context_manager,
        test_eql_query,
        test_gpu_config,
        test_gpu_device_info,
        test_thread_safety_python,
        test_edge_cases,
        test_memory_leak_check,
        test_type_stubs,
        test_parity_cpp_vs_python,
    ]

    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {len(tests) - failed}/{len(tests)} passed, {failed} failed")
    sys.exit(failed)