"""Exhaustive tests for the newly exposed Python surface.

Covers every symbol added when the bindings were widened to match the C++ API:
the core gaps, the GPU control surface, the EQL AST, WAL replay, index
snapshots, and the modern Arena/Engine/Accelerator wrappers.

Tests are adversarial on purpose: they assert on error paths, lifecycle
violations, boundary values, and concurrency, not only on the happy path. GPU
tests skip cleanly when no device is present rather than failing.
"""

from __future__ import annotations

import enum
import os
import tempfile
import threading
import warnings

import elips
import pytest
from elips import _core as core, query as eql

requires_gpu = pytest.mark.skipif(
    not getattr(elips, "_has_gpu", False) or not elips.gpu_devices(),
    reason="no GPU device available",
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "db")


@pytest.fixture
def memdb():
    db = elips.open(":memory:", dimension=4)
    yield db
    db.abandon()


# ============================  identifiers  ============================


def test_generate_id_is_unique_and_valid():
    ids = {elips.generate_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(elips.is_valid_id(rid) for rid in ids)


def test_generated_id_round_trips_through_place(memdb):
    vault = memdb.vault("v")
    rid = elips.generate_id()
    assert vault.place([1.0, 0.0, 0.0, 0.0], {}, rid) == rid
    assert vault.fetch(rid) is not None


@pytest.mark.parametrize(
    "bad",
    ["", "not-an-id", "0" * 100, "☃", "019fb1cd-7f3", "  "],
)
def test_is_valid_id_rejects_malformed(bad):
    assert not elips.is_valid_id(bad)


def test_place_with_malformed_id_raises(memdb):
    with pytest.raises(elips.ElipsError):
        memdb.vault("v").place([1.0, 0.0, 0.0, 0.0], {}, "not-an-id")


# ==========================  vector helpers  ===========================


def test_normalize_produces_unit_length():
    out = elips.normalize([3.0, 4.0])
    assert out == pytest.approx((0.6, 0.8))
    assert elips.magnitude(out) == pytest.approx(1.0)


def test_normalize_zero_vector_is_unchanged():
    assert elips.normalize([0.0, 0.0, 0.0]) == pytest.approx((0.0, 0.0, 0.0))


def test_magnitude_matches_manual_computation():
    values = [1.5, -2.5, 0.25]
    expected = sum(v * v for v in values) ** 0.5
    assert elips.magnitude(values) == pytest.approx(expected)


def test_normalize_is_idempotent():
    once = elips.normalize([5.0, 12.0])
    assert elips.normalize(once) == pytest.approx(once)


# =============================  filters  ===============================


@pytest.mark.parametrize(
    ("op", "value", "payload", "expected"),
    [
        ("eq", 5, {"n": 5}, True),
        ("eq", 5, {"n": 6}, False),
        ("ne", 5, {"n": 6}, True),
        ("lt", 5, {"n": 4}, True),
        ("lt", 5, {"n": 5}, False),
        ("le", 5, {"n": 5}, True),
        ("gt", 5, {"n": 6}, True),
        ("ge", 5, {"n": 5}, True),
        ("gte", 5, {"n": 5}, True),
    ],
)
def test_filter_compare_string_operators(op, value, payload, expected):
    assert elips.Filter.compare("n", op, value).matches(payload) is expected


def test_filter_compare_accepts_enum():
    flt = elips.Filter.compare("n", elips.Comparator.ge, 10)
    assert flt.matches({"n": 10})
    assert not flt.matches({"n": 9})


def test_filter_compare_rejects_unknown_operator():
    with pytest.raises(ValueError):
        elips.Filter.compare("n", "approximately", 1)


def test_filter_in_set_and_has_substring():
    assert elips.Filter.in_set("t", ["a", "b"]).matches({"t": "b"})
    assert not elips.Filter.in_set("t", ["a", "b"]).matches({"t": "c"})
    assert elips.Filter.has_substring("body", "ell").matches({"body": "hello"})
    assert not elips.Filter.has_substring("body", "zzz").matches({"body": "hello"})


def test_filter_matches_missing_key_is_false():
    assert not elips.Filter.compare("absent", "eq", 1).matches({"present": 1})


def test_empty_filter_matches_everything():
    empty = elips.Filter()
    assert empty.matches_all()
    assert empty.matches({})
    assert empty.matches({"anything": "at all"})


def test_filter_combinators_compose():
    left = elips.Filter.compare("year", "ge", 2023)
    right = elips.Filter.in_set("tier", ["pro"])
    both = left.and_(right)
    either = left.or_(right)

    assert both.matches({"year": 2024, "tier": "pro"})
    assert not both.matches({"year": 2024, "tier": "free"})
    assert either.matches({"year": 2024, "tier": "free"})
    assert not either.matches({"year": 2020, "tier": "free"})
    assert elips.Filter.not_(left).matches({"year": 2020})


def test_exact_constraints_reports_index_accelerable_predicates():
    assert elips.Filter.in_set("t", ["a"]).exact_constraints() == [("t", ["a"])]
    # A range predicate cannot be answered from the equality index.
    assert elips.Filter.compare("n", "gt", 5).exact_constraints() is None


def test_filter_matches_agrees_with_server_side_filtering(memdb):
    vault = memdb.vault("v")
    payloads = [{"n": i, "even": i % 2 == 0} for i in range(20)]
    for i, payload in enumerate(payloads):
        vault.place([float(i), 0.0, 0.0, 0.0], payload)

    flt = elips.Filter.compare("n", "ge", 10).and_(
        elips.Filter.compare("even", "eq", True)
    )
    expected = sum(1 for payload in payloads if flt.matches(payload))
    assert len(vault.scan(where=flt, limit=100)) == expected


# ==========================  graph params  =============================


def test_graph_params_exposes_compaction_ratio():
    params = elips.GraphParams(
        max_connections=8, ef_construction=100, ef_search=25, compaction_ratio=0.4
    )
    assert (params.max_connections, params.ef_search) == (8, 25)
    assert params.compaction_ratio == pytest.approx(0.4)
    params.compaction_ratio = 0.0
    assert params.compaction_ratio == 0.0


def test_compaction_ratio_zero_disables_auto_compaction():
    config = (
        elips.Config()
        .dimension(4)
        .graph_params(elips.GraphParams(compaction_ratio=0.0))
    )
    db = elips.open_with_config(":memory:", config)
    vault = db.vault("v")
    ids = [vault.place([float(i), 0.0, 0.0, 0.0]) for i in range(10)]
    for rid in ids[:8]:
        vault.erase(rid)
    # 80% tombstones and still no rebuild, because the trigger is disabled.
    assert vault.pending_removals == 8
    vault.vacuum()
    assert vault.pending_removals == 0
    db.abandon()


def test_compaction_ratio_triggers_rebuild():
    config = (
        elips.Config()
        .dimension(4)
        .graph_params(elips.GraphParams(compaction_ratio=0.25))
    )
    db = elips.open_with_config(":memory:", config)
    vault = db.vault("v")
    ids = [vault.place([float(i), 0.0, 0.0, 0.0]) for i in range(20)]
    for rid in ids:
        vault.erase(rid)
        assert vault.pending_removals <= 6  # never far past the 25% threshold
    db.abandon()


# ==============================  vault  ================================


def test_records_snapshot_matches_scan(memdb):
    vault = memdb.vault("v")
    for i in range(5):
        vault.place([float(i), 1.0, 0.0, 0.0], {"i": i})
    records = vault.records()
    assert len(records) == 5
    assert {r["id"] for r in records} == {r["id"] for r in vault.scan(limit=100)}
    assert all(set(r) >= {"id", "vector", "data"} for r in records)


def test_records_returns_a_copy_not_a_live_view(memdb):
    vault = memdb.vault("v")
    vault.place([1.0, 0.0, 0.0, 0.0])
    snapshot = vault.records()
    vault.place([0.0, 1.0, 0.0, 0.0])
    assert len(snapshot) == 1
    assert len(vault.records()) == 2


def test_read_only_toggle_blocks_then_allows_writes(memdb):
    vault = memdb.vault("v")
    assert not vault.read_only
    vault.set_read_only(True)
    assert vault.read_only

    with pytest.raises(elips.StorageError):
        vault.place([1.0, 0.0, 0.0, 0.0])

    vault.set_read_only(False)
    assert isinstance(vault.place([1.0, 0.0, 0.0, 0.0]), str)


def test_read_only_still_permits_reads(memdb):
    vault = memdb.vault("v")
    rid = vault.place([1.0, 0.0, 0.0, 0.0])
    vault.set_read_only(True)
    assert vault.fetch(rid) is not None
    assert vault.count() == 1
    assert len(vault.seek([1.0, 0.0, 0.0, 0.0], 1)) == 1


def test_pending_removals_and_vacuum(db_path):
    config = (
        elips.Config()
        .dimension(4)
        .graph_params(elips.GraphParams(compaction_ratio=0.0))
    )
    db = elips.open_with_config(db_path, config)
    vault = db.vault("v")
    ids = [vault.place([float(i), 0.0, 0.0, 0.0]) for i in range(10)]
    assert vault.pending_removals == 0
    for rid in ids[:4]:
        vault.erase(rid)
    assert vault.pending_removals == 4
    vault.vacuum()
    assert vault.pending_removals == 0
    assert vault.count() == 6
    db.close()


def test_vacuum_preserves_searchability(memdb):
    vault = memdb.vault("v")
    keep = [vault.place([float(i), 1.0, 0.0, 0.0]) for i in range(10)]
    drop = [vault.place([0.0, 0.0, float(i), 1.0]) for i in range(10)]
    for rid in drop:
        vault.erase(rid)
    vault.vacuum()

    hits = vault.seek([0.0, 1.0, 0.0, 0.0], 10)
    assert len(hits) == 10
    assert {hit.id for hit in hits} <= set(keep)


# ===========================  lifecycle  ===============================


def test_writes_after_close_raise_and_do_not_persist(db_path):
    db = elips.open(db_path, dimension=4)
    vault = db.vault("v")
    vault.place([1.0, 0.0, 0.0, 0.0])
    db.close()

    assert vault.sealed
    with pytest.raises(elips.StorageError):
        vault.place([0.0, 1.0, 0.0, 0.0])
    with pytest.raises(elips.StorageError):
        vault.place_many([])

    reopened = elips.open(db_path)
    assert reopened.vault("v").count() == 1
    reopened.abandon()


def test_vault_created_after_close_is_already_sealed(db_path):
    db = elips.open(db_path, dimension=4)
    db.vault("v").place([1.0, 0.0, 0.0, 0.0])
    db.close()
    late = db.vault("late")
    assert late.sealed
    with pytest.raises(elips.StorageError):
        late.place([1.0, 0.0, 0.0, 0.0])


def test_transaction_commit_after_close_raises(db_path):
    db = elips.open(db_path, dimension=4)
    txn = db.begin_transaction()
    txn.vault("v").place([1.0, 0.0, 0.0, 0.0])
    db.close()
    with pytest.raises(elips.StorageError):
        txn.commit()


def test_close_is_idempotent(db_path):
    db = elips.open(db_path, dimension=4)
    db.vault("v").place([1.0, 0.0, 0.0, 0.0])
    db.close()
    db.close()
    assert db.closed


def test_database_path_and_persistence_flags(db_path):
    db = elips.open(db_path, dimension=4)
    assert db.path == db_path
    assert db.persistent
    assert not db.closed
    db.close()
    assert db.closed

    mem = elips.open(":memory:", dimension=4)
    assert mem.path == ":memory:"
    assert not mem.persistent
    mem.abandon()


# =========================  transactions  ==============================


def test_transaction_commit_applies_whole_batch(memdb):
    with memdb.begin_transaction() as txn:
        vault = txn.vault("v")
        for i in range(10):
            vault.place([float(i), 0.0, 0.0, 0.0])
    assert memdb.vault("v").count() == 10


def test_transaction_rollback_applies_nothing(memdb):
    txn = memdb.begin_transaction()
    txn.vault("v").place([1.0, 0.0, 0.0, 0.0])
    txn.rollback()
    assert memdb.vault("v").count() == 0


def test_commit_on_read_only_vault_is_atomic(memdb):
    vault = memdb.vault("v")
    survivor = vault.place([9.0, 9.0, 9.0, 9.0])
    vault.set_read_only(True)

    txn = memdb.begin_transaction()
    tv = txn.vault("v")
    for i in range(5):
        tv.place([float(i), 0.0, 0.0, 0.0])

    with pytest.raises(elips.StorageError):
        txn.commit()

    vault.set_read_only(False)
    assert vault.count() == 1
    assert vault.fetch(survivor) is not None


def test_transaction_erase_round_trip(memdb):
    vault = memdb.vault("v")
    rid = vault.place([1.0, 0.0, 0.0, 0.0])
    with memdb.begin_transaction() as txn:
        txn.vault("v").erase(rid)
    assert vault.fetch(rid) is None


# ==============================  config  ===============================


def test_config_introspection_round_trips():
    config = (
        elips.Config()
        .dimension(16)
        .metric("euclidean")
        .index("exact")
        .durability("paranoid")
        .access_mode("read_only")
        .segmented_storage(False)
        .metadata_acceleration(False)
    )
    assert config.dimension_val == 16
    assert config.metric_val == "euclidean"
    assert config.index_val == "exact"
    assert config.durability_val == "paranoid"
    assert config.access_mode_val == "read_only"
    assert not config.segmented_storage_enabled
    assert not config.metadata_acceleration_enabled


@pytest.mark.parametrize("level", ["paranoid", "standard", "relaxed", "ephemeral"])
def test_every_durability_level_round_trips(level):
    assert elips.Config().durability(level).durability_val == level


def test_pending_local_embedder_introspection():
    config = elips.Config().dimension(8)
    assert not config.has_pending_local_text_embedder
    assert config.local_text_embedder_config is None

    config.local_text_embedder(elips.LocalEmbedderConfig(dimension=8))
    assert config.has_pending_local_text_embedder
    assert config.local_text_embedder_config.dimension == 8


def test_describe_local_embedder_resolves_without_instantiating():
    info = elips.describe_local_embedder(
        elips.LocalEmbedderConfig(model="default", revision="v1", dimension=32)
    )
    assert info.dimension == 32
    assert info.kind == elips.TextEmbedderKind.local_builtin
    assert not info.loaded
    assert info.fingerprint


def test_describe_local_embedder_uses_fallback_dimension():
    info = elips.describe_local_embedder(
        elips.LocalEmbedderConfig(), fallback_dimension=64
    )
    assert info.dimension == 64


# ============================  EQL AST  ================================


def test_parse_seek_statement_exposes_the_tree():
    stmt = elips.parse_eql("seek in docs nearest $q top 7 where year >= 2023 yield")
    assert isinstance(stmt, eql.SearchStatement)
    assert stmt.vault == "docs"
    assert stmt.top == 7
    assert stmt.query.binding == "q"
    assert not stmt.query.literal
    assert stmt.where.matches({"year": 2024})
    assert not stmt.where.matches({"year": 2020})


def test_parse_fetch_and_erase_statements():
    fetched = elips.parse_eql('fetch from v id "abc" yield')
    assert isinstance(fetched, eql.FetchStatement)
    assert (fetched.vault, fetched.id) == ("v", "abc")

    erased = elips.parse_eql('erase from v id "abc"')
    assert isinstance(erased, eql.DeleteStatement)
    assert (erased.vault, erased.id) == ("v", "abc")


def test_parse_rejects_invalid_eql():
    for bad in ["", "not eql at all", "seek in", "seek in v nearest"]:
        with pytest.raises(elips.ParseError):
            elips.parse_eql(bad)


def test_parse_agrees_with_validate():
    good = "seek in docs nearest $q top 1 yield"
    assert elips.validate_eql(good) is None
    assert isinstance(elips.parse_eql(good), eql.SearchStatement)


def test_parsed_filter_is_usable_against_the_database(memdb):
    vault = memdb.vault("docs")
    for year in (2020, 2023, 2025):
        vault.place([float(year % 10), 0.0, 0.0, 0.0], {"year": year})

    stmt = elips.parse_eql("seek in docs nearest $q top 5 where year >= 2023 yield")
    matched = vault.scan(where=stmt.where, limit=100)
    assert {int(row["data"]["year"]) for row in matched} == {2023, 2025}


def test_ast_nodes_are_mutable_for_query_rewriting():
    stmt = elips.parse_eql("seek in docs nearest $q top 100 yield")
    stmt.top = 10
    stmt.vault = "other"
    assert (stmt.top, stmt.vault) == (10, "other")


# ============================  WAL replay  =============================


def test_replay_wal_reports_acknowledged_writes(db_path):
    db = elips.open(db_path, dimension=4, index="exact")
    vault = db.vault("v")
    first = vault.place([1.0, 0.0, 0.0, 0.0], {"k": "a"})
    second = vault.place([0.0, 1.0, 0.0, 0.0])
    vault.erase(second)
    db.abandon()

    entries = elips.replay_wal(os.path.join(db_path, "wal.log"))
    assert [e.op for e in entries] == [
        elips.WalOp.insert,
        elips.WalOp.insert,
        elips.WalOp.erase,
    ]
    assert entries[0].id == first
    assert entries[0].vault == "v"
    assert entries[0].data == {"k": "a"}
    assert entries[0].vector == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert entries[2].vector == ()


def test_replay_wal_drops_uncommitted_transaction(db_path):
    db = elips.open(db_path, dimension=4)
    committed = db.vault("v").place([1.0, 0.0, 0.0, 0.0])
    txn = db.begin_transaction()
    txn.vault("v").place([0.0, 1.0, 0.0, 0.0])
    txn.rollback()
    db.abandon()

    entries = elips.replay_wal(os.path.join(db_path, "wal.log"))
    assert [e.id for e in entries] == [committed]


def test_replay_wal_survives_a_truncated_tail(db_path):
    db = elips.open(db_path, dimension=4)
    for i in range(5):
        db.vault("v").place([float(i), 0.0, 0.0, 0.0])
    db.abandon()

    wal = os.path.join(db_path, "wal.log")
    intact = len(elips.replay_wal(wal))
    with open(wal, "r+b") as handle:
        handle.truncate(os.path.getsize(wal) - 7)

    truncated = elips.replay_wal(wal)
    assert 0 <= len(truncated) < intact


def test_replay_wal_on_missing_or_empty_file(tmp_path):
    assert elips.replay_wal(str(tmp_path / "absent.log")) == []
    empty = tmp_path / "empty.log"
    empty.write_bytes(b"")
    assert elips.replay_wal(str(empty)) == []


def test_replay_wal_rejects_garbage_without_raising(tmp_path):
    junk = tmp_path / "junk.log"
    junk.write_bytes(bytes(range(256)) * 32)
    assert elips.replay_wal(str(junk)) == []


def test_checkpoint_truncates_the_wal(db_path):
    db = elips.open(db_path, dimension=4)
    db.vault("v").place([1.0, 0.0, 0.0, 0.0])
    wal = os.path.join(db_path, "wal.log")
    assert elips.replay_wal(wal)
    db.checkpoint()
    assert elips.replay_wal(wal) == []
    db.abandon()


# =========================  index snapshots  ===========================


def test_index_snapshot_types_are_constructible():
    """Snapshots are engine internals, reachable only through the seam.

    Nothing in the binding returns one, which is why they are demoted rather
    than rehomed: they were only ever constructible placeholders.
    """

    snapshot = core.IndexSnapshot()
    assert len(snapshot) == 0
    snapshot.dimension = 8
    snapshot.kind = core.IndexSnapshotKind.graph
    snapshot.metric = elips.Metric.cosine
    snapshot.vectors = [1.0, 2.0]
    assert snapshot.dimension == 8
    assert snapshot.kind == core.IndexSnapshotKind.graph
    assert snapshot.ivf is None and snapshot.pq is None

    ivf = core.IvfSnapshot()
    ivf.n_lists, ivf.n_probe = 16, 4
    snapshot.ivf = ivf
    assert snapshot.ivf.n_lists == 16

    pq = core.PqSnapshot()
    pq.pq_dim, pq.pq_bits = 8, 6
    snapshot.pq = pq
    assert (snapshot.pq.pq_dim, snapshot.pq.pq_bits) == (8, 6)


# ==============================  GPU  ==================================


def test_gpu_device_enumeration_is_always_callable():
    devices = elips.gpu_devices()
    assert isinstance(devices, list)
    for device in devices:
        assert device.name and device.backend
        assert device.total_memory_bytes > 0


def test_cpu_fallback_and_runtime_info_are_always_available():
    assert elips.gpu_cpu_fallback_info().name
    assert elips.gpu_runtime_device_info().name


def test_gpu_error_message_covers_every_code():
    for name in dir(elips.GpuError):
        if name.startswith("_"):
            continue
        value = getattr(elips.GpuError, name)
        if isinstance(value, elips.GpuError):
            assert elips.gpu_error_message(value)


@requires_gpu
def test_can_fit_index_scales_with_request_size():
    device = elips.gpu_devices()[0]
    assert elips.gpu_can_fit_index(device, 1_000, 64)
    # An index far larger than any real device cannot fit.
    assert not elips.gpu_can_fit_index(device, 10**12, 4096)


@requires_gpu
def test_gpu_select_returns_a_usable_handle():
    device = elips.gpu_select()
    assert device is not None
    with device:
        assert device.available
        assert device.backend
        assert device.device_info.name
        device.synchronize()
        assert device.idle
    assert device.closed


@requires_gpu
def test_gpu_distance_kernel_matches_the_cpu():
    corpus = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]]
    queries = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    with elips.gpu_select() as device:
        rows = device.compute_distances(queries, corpus, metric="cosine")

    assert len(rows) == len(queries)
    for query, row in zip(queries, rows, strict=False):
        assert len(row) == len(corpus)
        for entry, expected in zip(
            row, (elips.distance("cosine", query, c) for c in corpus), strict=False
        ):
            assert entry == pytest.approx(expected, abs=1e-4)


@requires_gpu
def test_gpu_top_k_selects_the_nearest():
    corpus = [[float(i), 1.0] for i in range(16)]
    with elips.gpu_select() as device:
        rows = device.compute_distances([[0.0, 1.0]], corpus, metric="euclidean")
        indices, values = device.top_k(rows, k=3)

    assert indices[0][0] == 0
    assert values[0] == sorted(values[0])


@requires_gpu
def test_gpu_kernels_validate_their_inputs():
    with elips.gpu_select() as device:
        with pytest.raises(elips.DimensionMismatch):
            device.compute_distances([[1.0, 2.0]], [[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            device.top_k([[1.0, 2.0]], k=99)
        with pytest.raises(ValueError):
            device.compute_distances([], [[1.0]])


@requires_gpu
def test_closed_gpu_handle_refuses_work():
    device = elips.gpu_select()
    device.close()
    device.close()  # idempotent
    assert device.closed
    with pytest.raises(elips.StorageError):
        device.compute_distances([[1.0]], [[1.0]])
    with pytest.raises(elips.StorageError):
        device.synchronize()


@requires_gpu
def test_gpu_memory_accounting_is_consistent():
    with elips.gpu_select() as device:
        device.memory.initialize(32 * 1024 * 1024)
        assert device.memory.bytes_used == 0
        assert device.memory.bytes_available == 32 * 1024 * 1024
        assert device.memory.peak_bytes_used >= device.memory.bytes_used


@requires_gpu
def test_gpu_profiler_records_and_clears():
    with elips.gpu_select() as device:
        assert device.profiler.total_launches == 0
        device.profiler.record("my_kernel", 1234, 42)
        timings = device.profiler.recent_timings()
        assert device.profiler.total_launches == 1
        assert timings[-1].kernel_name == "my_kernel"
        assert timings[-1].duration_us == 1234
        assert timings[-1].work_items == 42
        device.profiler.clear()
        assert device.profiler.total_launches == 0


def test_batch_stats_defaults():
    stats = elips.BatchStats()
    assert stats.queries_coalesced == 0
    assert stats.kernel_launches == 0
    assert stats.avg_batch_size == pytest.approx(0.0)


# =====================  modern arena / engine  =========================


def test_arena_health_reports_live_and_tombstones():
    engine = elips.connect(":memory:", dimension=4)
    arena = engine.arena("docs")
    keys = [arena.write(vector=[float(i), 1.0, 0.0, 0.0]) for i in range(20)]
    health = arena.health()
    assert health.name == "docs"
    assert health.live == 20
    assert health.pending_removals == 0
    assert health.dimension == 4
    assert health.metric == "cosine"
    assert not health.read_only and not health.sealed
    assert health.tombstone_ratio == pytest.approx(0.0)

    arena.discard(keys[:2])
    assert arena.health().tombstone_ratio == pytest.approx(2 / 20)
    engine.close()
    assert arena.health().sealed


def test_arena_health_ratio_on_empty_arena():
    engine = elips.connect(":memory:", dimension=4)
    assert engine.arena("empty").health().tombstone_ratio == 0.0
    engine.close()


def test_arena_freeze_and_thaw():
    engine = elips.connect(":memory:", dimension=4)
    arena = engine.arena("docs")
    arena.freeze()
    assert arena.read_only
    with pytest.raises(elips.StorageError):
        arena.write(vector=[1.0, 0.0, 0.0, 0.0])
    arena.freeze(False)
    assert not arena.read_only
    assert isinstance(arena.write(vector=[1.0, 0.0, 0.0, 0.0]), str)
    engine.close()


def test_arena_vacuum_and_rebuild_preserve_data():
    engine = elips.connect(":memory:", dimension=4)
    arena = engine.arena("docs")
    keys = [arena.write(vector=[float(i), 1.0, 0.0, 0.0]) for i in range(10)]
    arena.discard(keys[:3])
    arena.vacuum()
    assert arena.pending_removals == 0
    arena.rebuild()
    assert arena.count() == 7
    assert len(arena.probe([0.0, 1.0, 0.0, 0.0], top=7)) == 7
    engine.close()


def test_engine_vacuum_covers_every_arena():
    engine = elips.connect(":memory:", dimension=4)
    for name in ("a", "b"):
        arena = engine.arena(name)
        keys = [arena.write(vector=[float(i), 1.0, 0.0, 0.0]) for i in range(20)]
        arena.discard(keys[:2])
    engine.vacuum()
    assert engine.arena("a").pending_removals == 0
    assert engine.arena("b").pending_removals == 0
    assert sorted(engine.vault_names()) == ["a", "b"]
    engine.close()


def test_engine_pending_writes_reads_the_wal():
    path = tempfile.mkdtemp()
    engine = elips.connect(path, dimension=4)
    arena = engine.arena("docs")
    key = arena.write(vector=[1.0, 0.0, 0.0, 0.0], meta={"k": "v"})
    arena.discard([key])

    records = engine.pending_writes()
    assert [r.op for r in records] == ["insert", "erase"]
    assert records[0].arena == "docs"
    assert records[0].key == key
    assert records[0].meta == {"k": "v"}
    assert not records[0].is_delete
    assert records[1].is_delete
    engine.close()


def test_engine_pending_writes_is_empty_for_memory_databases():
    engine = elips.connect(":memory:", dimension=4)
    engine.arena("docs").write(vector=[1.0, 0.0, 0.0, 0.0])
    assert engine.pending_writes() == []
    engine.close()


def test_wal_record_from_entry_matches_low_level(db_path):
    db = elips.open(db_path, dimension=4)
    key = db.vault("v").place([1.0, 0.0, 0.0, 0.0], {"n": 7})
    db.abandon()

    entry = elips.replay_wal(os.path.join(db_path, "wal.log"))[0]
    record = elips.WalRecord.from_entry(entry)
    assert record.op == entry.op.name
    assert record.arena == "v"
    assert record.key == key
    assert record.meta == {"n": 7}


# ====================  modern accelerator facade  ======================


def test_accelerators_listing_is_always_safe():
    specs = elips.accelerators()
    assert isinstance(specs, list)
    for spec in specs:
        assert spec.name and spec.backend
        assert spec.memory_bytes > 0
        assert spec.memory_gb == pytest.approx(spec.memory_bytes / 1024**3)
        assert isinstance(spec.can_fit(1000, 64), bool)


def test_accelerator_returns_none_or_a_handle():
    gpu = elips.accelerator()
    if gpu is None:
        pytest.skip("no GPU device available")
    with gpu:
        assert gpu.backend
        assert gpu.spec.name
        assert not gpu.closed
    assert gpu.closed
    assert "closed" in repr(gpu)


@requires_gpu
def test_accelerator_search_finds_the_nearest_neighbour():
    corpus = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    with elips.accelerator() as gpu:
        gpu.reserve(16 * 1024 * 1024)
        indices, values = gpu.search([[1.0, 0.0]], corpus, top=2)
        assert indices[0][0] == 0
        assert values[0][0] == pytest.approx(0.0, abs=1e-5)

        rows = gpu.distances([[1.0, 0.0]], corpus)
        assert gpu.nearest(rows, top=1)[0][0][0] == 0

        gpu.synchronize()
        used, available, peak = gpu.memory_usage()
        assert used >= 0 and available >= 0 and peak >= used
        assert isinstance(gpu.kernel_timings(), list)


@requires_gpu
def test_accelerator_raw_handle_is_reachable():
    with elips.accelerator() as gpu:
        assert isinstance(gpu.raw, elips.GpuDevice)
        assert gpu.raw.backend == gpu.backend


# ===========================  concurrency  =============================


def test_concurrent_writers_and_readers_on_one_vault(memdb):
    vault = memdb.vault("v")
    for i in range(32):
        vault.place([float(i), 0.0, 0.0, 0.0])

    errors: list[BaseException] = []
    stop = threading.Event()

    def writer(worker: int) -> None:
        try:
            for i in range(100):
                vault.place([float(worker * 1000 + i), 1.0, 0.0, 0.0])
        except BaseException as exc:  # noqa: BLE001 # pragma: no cover - failure path
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                vault.seek([1.0, 0.0, 0.0, 0.0], 5)
                vault.count()
                vault.scan(limit=4)
        except BaseException as exc:  # noqa: BLE001 # pragma: no cover - failure path
            errors.append(exc)

    writers = [threading.Thread(target=writer, args=(w,)) for w in range(3)]
    readers = [threading.Thread(target=reader) for _ in range(3)]
    for thread in writers + readers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    for thread in readers:
        thread.join()

    assert not errors
    assert vault.count() == 32 + 300


def test_concurrent_transactions_do_not_interleave(memdb):
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            for batch in range(10):
                with memdb.begin_transaction() as txn:
                    vault = txn.vault("v")
                    for i in range(5):
                        vault.place([float(worker_id), float(batch), float(i), 0.0])
        except BaseException as exc:  # noqa: BLE001 # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert memdb.vault("v").count() == 4 * 10 * 5


def test_concurrent_vault_creation_is_safe(memdb):
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            for i in range(15):
                memdb.vault(f"vault_{i}").place([float(worker_id), 0.0, 0.0, 0.0])
        except BaseException as exc:  # noqa: BLE001 # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(memdb.list_vaults()) == 15
    for i in range(15):
        assert memdb.vault(f"vault_{i}").count() == 6


# ==========================  quantization  =============================


def _gaussian_rows(count: int, dim: int, seed: int = 11) -> list[list[float]]:
    import random

    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(count)]


def test_quant_params_defaults_and_repr():
    params = elips.QuantParams()
    assert params.codec == "none"
    assert params.pq_dim == 0
    assert params.pq_bits == 8
    assert "QuantParams" in repr(params)

    configured = elips.QuantParams(codec="pq", pq_dim=16, pq_bits=6, train_iters=4)
    assert configured.codec == "pq"
    assert configured.pq_dim == 16
    assert configured.pq_bits == 6
    assert configured.train_iters == 4
    assert configured.codec_enum == elips.Codec.pq


def test_quant_params_reports_code_width_before_training():
    # Lets a caller size a deployment without ingesting anything first.
    assert elips.QuantParams(codec="pq", pq_dim=24).code_bytes(768) == 24
    assert elips.QuantParams(codec="sq8").code_bytes(768) == 768
    assert elips.QuantParams(codec="none").code_bytes(768) == 0
    # 768 floats is 3072 bytes, so 24 subspaces is 128x.
    assert (768 * 4) // elips.QuantParams(codec="pq", pq_dim=24).code_bytes(768) == 128


@pytest.mark.parametrize("codec", ["pq", "opq", "sq8"])
def test_quant_params_accepts_every_codec(codec):
    params = elips.QuantParams(codec=codec)
    assert params.codec == codec
    params.validate(64)


def test_quant_params_rejects_unknown_codec_and_bad_geometry():
    with pytest.raises(elips.ConfigError):
        elips.QuantParams(codec="lz4")
    with pytest.raises(elips.ConfigError):
        elips.QuantParams(codec="pq", pq_dim=7).validate(64)  # 64 % 7 != 0
    with pytest.raises(elips.ConfigError):
        elips.QuantParams(codec="pq", pq_bits=9).validate(64)
    with pytest.raises(elips.ConfigError):
        elips.QuantParams(codec="pq", pq_bits=3).validate(64)


def test_config_carries_quantization():
    params = elips.QuantParams(codec="sq8")
    config = elips.Config().dimension(32).quantization(params)
    assert config.has_quantization
    assert config.quantization_val.codec == "sq8"
    assert not elips.Config().dimension(32).has_quantization


def test_open_accepts_a_bare_codec_name():
    with elips.open(":memory:", dimension=8, quantization="sq8") as db:
        assert db.config.has_quantization
        assert db.config.quantization_val.codec == "sq8"


def test_vault_quantize_compresses_and_labels_results():
    params = elips.QuantParams(codec="pq", pq_dim=8, train_iters=4)
    with elips.open(
        ":memory:", dimension=32, metric="euclidean", quantization=params
    ) as db:
        vault = db.vault("main")
        rows = _gaussian_rows(300, 32)
        for row in rows:
            vault.place(row)

        assert not vault.quantized
        assert vault.codec == "none"
        assert vault.info().codec == "none"
        assert vault.info().code_bytes == 0
        assert vault.info().compression_ratio == 1.0

        vault.quantize()

        assert vault.quantized
        assert vault.codec == "pq"
        info = vault.info()
        assert info.codec == "pq"
        assert info.code_bytes == 8
        # 32 floats down to 8 bytes.
        assert info.compression_ratio == 16.0
        assert info.count == 300

        hit = vault.seek(rows[0], top=1)[0]
        assert hit.approximate
        assert hit.codec == "pq"
        assert "approximate" in repr(hit)

        record = vault.records()[0]
        assert record["approximate"]
        assert record["codec"] == "pq"


def test_unquantized_vault_reports_exact_results():
    with elips.open(":memory:", dimension=4, metric="euclidean") as db:
        vault = db.vault("main")
        vault.place([1.0, 0.0, 0.0, 0.0])
        hit = vault.seek([1.0, 0.0, 0.0, 0.0], top=1)[0]
        assert not hit.approximate
        assert hit.codec == "none"
        record = vault.records()[0]
        assert not record["approximate"]
        assert record["codec"] == "none"
        # An uncompressed vault still returns the exact stored vector.
        assert record["vector"] == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_quantize_rejects_invalid_states():
    params = elips.QuantParams(codec="pq", pq_dim=4, train_iters=3)
    with elips.open(
        ":memory:", dimension=16, metric="euclidean", quantization=params
    ) as db:
        vault = db.vault("main")
        # No codebook can be trained from nothing.
        with pytest.raises(elips.ConfigError):
            vault.quantize()

        for row in _gaussian_rows(60, 16):
            vault.place(row)
        vault.quantize()
        # Already compressed.
        with pytest.raises(elips.ConfigError):
            vault.quantize()


def test_quantize_requires_a_configured_codec():
    with elips.open(":memory:", dimension=8, metric="euclidean") as db:
        vault = db.vault("main")
        vault.place([1.0] * 8)
        with pytest.raises(elips.ConfigError):
            vault.quantize()


def test_database_quantize_all_covers_every_vault():
    params = elips.QuantParams(codec="sq8")
    with elips.open(
        ":memory:", dimension=8, metric="euclidean", quantization=params
    ) as db:
        for name in ("alpha", "beta"):
            vault = db.vault(name)
            for row in _gaussian_rows(40, 8, seed=hash(name) % 1000):
                vault.place(row)

        db.quantize_all()
        for name in ("alpha", "beta"):
            assert db.vault(name).quantized
            assert db.vault(name).info().code_bytes == 8


def test_inserts_after_quantization_are_encoded_on_arrival():
    params = elips.QuantParams(codec="sq8")
    with elips.open(
        ":memory:", dimension=8, metric="euclidean", quantization=params
    ) as db:
        vault = db.vault("main")
        for row in _gaussian_rows(40, 8):
            vault.place(row)
        vault.quantize()

        key = vault.place([0.5] * 8)
        record = vault.fetch(key)
        assert record is not None
        assert record["approximate"]
        assert record["codec"] == "sq8"


def test_quantized_vault_survives_reopen_on_disk():
    params = elips.QuantParams(codec="pq", pq_dim=8, train_iters=4)
    rows = _gaussian_rows(200, 32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "db")
        with elips.open(
            path, dimension=32, metric="euclidean", quantization=params
        ) as db:
            vault = db.vault("main")
            for row in rows:
                vault.place(row)
            db.quantize("main")
            first = [h.id for h in vault.seek(rows[0], top=5)]

        # The codebook is persisted with the snapshot, so the reopened vault is
        # still compressed and ranks identically.
        with elips.open(
            path, dimension=32, metric="euclidean", quantization=params
        ) as db:
            vault = db.vault("main")
            assert vault.quantized
            assert vault.codec == "pq"
            assert vault.info().count == len(rows)
            assert [h.id for h in vault.seek(rows[0], top=5)] == first


def test_quantized_vault_recall_is_useful():
    # Not a tight bound -- this asserts the codes carry real signal rather than
    # that PQ hits a particular recall on this fixture.
    params = elips.QuantParams(codec="sq8")
    rows = _gaussian_rows(200, 16)
    with elips.open(":memory:", dimension=16, metric="euclidean") as exact_db:
        exact = exact_db.vault("main")
        keys = [exact.place(row) for row in rows]
        truth = [h.id for h in exact.seek(rows[0], top=5)]

    with elips.open(
        ":memory:", dimension=16, metric="euclidean", quantization=params
    ) as db:
        vault = db.vault("main")
        for key, row in zip(keys, rows, strict=False):
            vault.place(row, id=key)
        vault.quantize()
        got = [h.id for h in vault.seek(rows[0], top=5)]

    assert len(set(truth) & set(got)) >= 4


def test_modern_arena_compress_and_health():
    engine = elips.connect(":memory:", dimension=8, quantization="sq8")
    try:
        arena = engine.arena("documents")
        for row in _gaussian_rows(40, 8):
            arena.write(vector=row)

        assert arena.health().codec == "none"
        assert arena.health().code_bytes == 0

        arena.compress()

        health = arena.health()
        assert health.codec == "sq8"
        assert health.code_bytes == 8

        hit = arena.probe([1.0] * 8, top=1)[0]
        assert hit.approximate
        assert hit.codec == "sq8"

        row = arena.sweep(limit=1)[0]
        assert row.approximate
        assert row.codec == "sq8"
    finally:
        engine.close()


def test_modern_connect_accepts_quant_params():
    engine = elips.connect(
        ":memory:", dimension=8, quantization=elips.QuantParams(codec="pq", pq_dim=4)
    )
    try:
        arena = engine.arena("documents")
        for row in _gaussian_rows(60, 8):
            arena.write(vector=row)
        arena.compress()
        assert arena.health().codec == "pq"
        assert arena.health().code_bytes == 4
    finally:
        engine.close()


def test_codec_enum_is_exported():
    assert elips.Codec.none is not None
    assert elips.Codec.pq != elips.Codec.opq
    assert {"none", "pq", "opq", "sq8"} <= set(elips.Codec.__members__)


# ==========================  stub coverage  ============================


def test_every_public_symbol_is_declared_in_the_stub():
    import ast
    import pathlib

    stub_path = pathlib.Path(core.__file__).with_name("_core.pyi")
    tree = ast.parse(stub_path.read_text())
    declared = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    declared |= {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    exported = {name for name in dir(core) if not name.startswith("_")}
    missing = exported - declared
    assert not missing, f"undeclared in _core.pyi: {sorted(missing)}"


def test_enum_members_do_not_leak_into_the_module_namespace():
    """``enum class`` members belong to their enum, not to ``elips._core``.

    ``export_values()`` is only appropriate for C-style unscoped enums. Calling
    it on a scoped enum publishes every member as a bare module attribute, so
    ``elips._core.none`` and ``elips._core.string`` become exported names that
    no stub declares.
    """

    leaked = {
        member
        for name in dir(core)
        if isinstance(getattr(core, name, None), type)
        and issubclass(getattr(core, name), enum.Enum)
        for member in getattr(core, name).__members__
        if hasattr(core, member)
    }
    assert not leaked, f"enum members leaked into elips._core: {sorted(leaked)}"


def test_facade_reexports_match_the_extension():
    """Every legacy name still resolves, demoted ones included.

    Demotion narrows what the package *advertises*; it does not remove
    anything yet. Warnings are suppressed here because this test is asserting
    exactly the back-compat the warnings announce -- `test_native_seam.py`
    covers the warning itself.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for name in elips.core.__all__:
            assert hasattr(elips, name), f"{name} missing from the elips namespace"
