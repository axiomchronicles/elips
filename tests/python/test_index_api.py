"""FAISS-style Index API (Phase 6.6).

Runnable directly: ``PYTHONPATH=bindings/python python3 tests/python/test_index_api.py``.
No pytest dependency, matching the repo's existing test style.
"""

from __future__ import annotations

import sys

import numpy as np

import elips

RNG = np.random.default_rng(1234)


def _data(n: int, d: int) -> np.ndarray:
    return RNG.standard_normal((n, d)).astype(np.float32)


def test_flat_shapes_and_dtypes() -> None:
    d, n, k = 16, 200, 10
    idx = elips.IndexFlat(d, metric="l2")
    assert idx.d == d
    assert idx.is_trained  # Flat needs no training
    assert idx.ntotal == 0
    xb = _data(n, d)
    idx.add(xb)
    assert idx.ntotal == n
    dists, labels = idx.search(xb[:5], k)
    assert dists.shape == (5, k)
    assert labels.shape == (5, k)
    assert dists.dtype == np.float32
    assert labels.dtype == np.int64
    # A vector searched against itself is its own nearest neighbour.
    assert labels[0, 0] == 0
    assert labels[3, 0] == 3


def test_one_d_query_is_single_row() -> None:
    d = 8
    idx = elips.IndexFlat(d, metric="l2")
    xb = _data(50, d)
    idx.add(xb)
    dists, labels = idx.search(xb[7], 3)  # bare (d,) vector
    assert dists.shape == (1, 3)
    assert labels[0, 0] == 7


def test_padding_when_k_exceeds_ntotal() -> None:
    d = 4
    idx = elips.IndexFlat(d, metric="l2")
    idx.add(_data(3, d))
    dists, labels = idx.search(_data(1, d), 5)
    assert labels.shape == (1, 5)
    # Only 3 real hits; the rest are -1 with +inf distance (FAISS semantics).
    assert (labels[0, :3] != -1).all()
    assert (labels[0, 3:] == -1).all()
    assert np.isinf(dists[0, 3:]).all()


def test_add_with_ids_roundtrip() -> None:
    d = 8
    idx = elips.IndexFlat(d, metric="l2")
    xb = _data(20, d)
    ids = np.arange(1000, 1000 + 20, dtype=np.int64)
    idx.add_with_ids(xb, ids)
    assert idx.ntotal == 20
    _, labels = idx.search(xb[:4], 1)
    # Nearest to each row is itself, returned under the caller's id.
    assert labels[0, 0] == 1000
    assert labels[3, 0] == 1003


def test_reset() -> None:
    d = 8
    idx = elips.IndexFlat(d, metric="l2")
    idx.add(_data(10, d))
    assert idx.ntotal == 10
    idx.reset()
    assert idx.ntotal == 0
    # Usable again after reset, with labels restarting from 0.
    xb = _data(5, d)
    idx.add(xb)
    assert idx.ntotal == 5
    _, labels = idx.search(xb[2], 1)
    assert labels[0, 0] == 2


def test_ivfflat_training_and_nprobe() -> None:
    d, nlist = 32, 16
    idx = elips.IndexIVFFlat(d, nlist, metric="l2")
    assert not idx.is_trained
    assert idx.nprobe == 1  # FAISS default
    xb = _data(2000, d)
    idx.train(xb)
    assert idx.is_trained
    idx.add(xb)
    assert idx.ntotal == 2000
    idx.nprobe = nlist  # exhaustive probing
    assert idx.nprobe == nlist
    # reset() keeps the trained centroids.
    idx.reset()
    assert idx.ntotal == 0
    assert idx.is_trained


def _recall_at_1(approx_labels: np.ndarray, truth_labels: np.ndarray) -> float:
    hits = (approx_labels[:, 0] == truth_labels[:, 0]).sum()
    return float(hits) / approx_labels.shape[0]


def test_ivfflat_recall_vs_flat() -> None:
    d, nlist, k = 32, 16, 10
    xb = _data(3000, d)
    xq = _data(100, d)

    flat = elips.IndexFlat(d, metric="l2")
    flat.add(xb)
    _, truth = flat.search(xq, k)

    ivf = elips.IndexIVFFlat(d, nlist, metric="l2")
    ivf.train(xb)
    ivf.add(xb)
    ivf.nprobe = nlist  # probe every list -> should match Flat closely
    _, approx = ivf.search(xq, k)

    recall = _recall_at_1(approx, truth)
    assert recall >= 0.95, f"IVFFlat@nprobe={nlist} recall@1 too low: {recall}"


def test_index_factory() -> None:
    d, k = 32, 5
    xb = _data(2000, d)
    idx = elips.index_factory(d, "IVF32,Flat")
    assert not idx.is_trained
    idx.train(xb)
    idx.add(xb)
    assert idx.ntotal == 2000
    idx.nprobe = 32
    dists, labels = idx.search(xb[:3], k)
    assert dists.shape == (3, k)
    assert labels.dtype == np.int64

    flat = elips.index_factory(d, "Flat")
    flat.add(xb)
    assert flat.ntotal == 2000


def main() -> int:
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {test.__name__}: {exc!r}")
    passed = len(tests) - failed
    print(f"\nResults: {passed}/{len(tests)} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
