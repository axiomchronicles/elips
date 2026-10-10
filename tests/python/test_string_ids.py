"""Chroma-style arbitrary string ``ids`` support (Phase 6.3).

ELIPS RecordIDs are 32-hex UUIDs; Chroma lets ``add(ids=["doc-1"])`` use any
string. These tests cover the Python Collection layer that maps arbitrary
string ids to stable native RecordIDs and round-trips the original string
through persisted metadata.

Runnable directly: ``PYTHONPATH=bindings/python python3 tests/python/test_string_ids.py``.
No pytest dependency, matching the repo's existing test style.
"""

from __future__ import annotations

import os
import sys
import tempfile

import elips

DIM = 3


def _client() -> elips.Client:
    return elips.Client(dimension=DIM, metric="cosine")


def test_add_get_roundtrip_string_ids() -> None:
    client = _client()
    docs = client.get_or_create_collection("docs")
    assigned = docs.add(
        ids=["doc-1", "doc-2"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        metadatas=[{"kind": "a"}, {"kind": "b"}],
    )
    assert assigned == ["doc-1", "doc-2"], assigned

    rows = docs.get(ids=["doc-1"])
    assert len(rows) == 1
    row = rows[0]
    assert row.key == "doc-1"
    # The reserved key must not leak into user-visible metadata.
    assert "__elips_id__" not in row.meta
    assert row.meta == {"kind": "a"}
    client.close()
    print("  PASS test_add_get_roundtrip_string_ids")


def test_query_returns_string_ids() -> None:
    client = _client()
    docs = client.get_or_create_collection("docs")
    docs.add(ids=["doc-1", "doc-2"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hits = docs.query(query_embeddings=[1.0, 0.0, 0.0], n_results=2)
    assert {h.key for h in hits} == {"doc-1", "doc-2"}, [h.key for h in hits]
    assert all("__elips_id__" not in h.meta for h in hits)
    assert hits[0].key == "doc-1"  # nearest to [1,0,0]
    client.close()
    print("  PASS test_query_returns_string_ids")


def test_delete_by_string_id() -> None:
    client = _client()
    docs = client.get_or_create_collection("docs")
    docs.add(ids=["doc-1", "doc-2"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert docs.count() == 2
    removed = docs.delete(ids=["doc-2"])
    assert removed == 1
    assert docs.count() == 1
    assert docs.get(ids=["doc-2"]) == []
    assert docs.get(ids=["doc-1"])[0].key == "doc-1"
    client.close()
    print("  PASS test_delete_by_string_id")


def test_upsert_overwrites_same_string_id() -> None:
    client = _client()
    docs = client.get_or_create_collection("docs")
    docs.add(ids=["doc-1"], embeddings=[[1.0, 0.0, 0.0]], metadatas=[{"v": 1}])
    docs.upsert(ids=["doc-1"], embeddings=[[0.0, 1.0, 0.0]], metadatas=[{"v": 2}])
    assert docs.count() == 1
    row = docs.get(ids=["doc-1"])[0]
    assert row.key == "doc-1"
    assert row.meta == {"v": 2}
    client.close()
    print("  PASS test_upsert_overwrites_same_string_id")


def test_reserved_key_raises() -> None:
    client = _client()
    docs = client.get_or_create_collection("docs")
    try:
        docs.add(
            ids=["doc-1"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[{"__elips_id__": "sneaky"}],
        )
        raise AssertionError("expected ValueError for reserved metadata key")
    except ValueError as error:
        assert "__elips_id__" in str(error)
    client.close()
    print("  PASS test_reserved_key_raises")


def test_auto_id_returns_native_hex() -> None:
    client = _client()
    docs = client.get_or_create_collection("docs")
    assigned = docs.add(embeddings=[[1.0, 0.0, 0.0]])
    native = assigned if isinstance(assigned, str) else assigned[0]
    assert elips.is_valid_id(native)
    # get/query echo the native hex id for auto-id records.
    row = docs.get(ids=[native])[0]
    assert row.key == native
    assert "__elips_id__" not in row.meta
    client.close()
    print("  PASS test_auto_id_returns_native_hex")


def test_persistent_roundtrip_across_reopen() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "strids")
        client = elips.PersistentClient(path, dimension=DIM, metric="cosine")
        docs = client.get_or_create_collection("docs")
        docs.add(
            ids=["doc-1", "doc-2"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            metadatas=[{"kind": "a"}, {"kind": "b"}],
        )
        client.close()

        reopened = elips.PersistentClient(path, dimension=DIM, metric="cosine")
        docs2 = reopened.get_or_create_collection("docs")
        row = docs2.get(ids=["doc-1"])[0]
        assert row.key == "doc-1"
        assert row.meta == {"kind": "a"}
        hits = docs2.query(query_embeddings=[0.0, 1.0, 0.0], n_results=1)
        assert hits[0].key == "doc-2"
        reopened.close()
    print("  PASS test_persistent_roundtrip_across_reopen")


TESTS = [
    test_add_get_roundtrip_string_ids,
    test_query_returns_string_ids,
    test_delete_by_string_id,
    test_upsert_overwrites_same_string_id,
    test_reserved_key_raises,
    test_auto_id_returns_native_hex,
    test_persistent_roundtrip_across_reopen,
]


def main() -> int:
    print("test_string_ids:")
    for test in TESTS:
        test()
    print(f"OK ({len(TESTS)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
