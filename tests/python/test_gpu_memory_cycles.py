"""Phase 8 §7.2 hardening: Metal GPU create/destroy cycles don't leak or crash.

Repeatedly builds and tears down a Metal-backed engine and dispatches the
exact brute-force k-NN (`bf_knn`) on the GPU. The invariants:

* every cycle completes without crashing (catches use-after-free / double-free
  in GPU buffer and command-queue teardown);
* results stay correct on every cycle (catches silent state corruption);
* resident memory plateaus — process high-water RSS must not keep climbing
  linearly with the cycle count (catches an unbounded per-cycle leak).

Skipped entirely when no runtime Metal device is present.
"""

from __future__ import annotations

import resource

import elips
import numpy as np
import pytest


def _has_metal() -> bool:
    return bool(getattr(elips, "_has_gpu", False)) and any(
        d.backend == "metal" for d in elips.gpu_devices()
    )


pytestmark = pytest.mark.skipif(not _has_metal(), reason="no runtime Metal device")

_DIM = 16
_NB = 2000
_RNG = np.random.RandomState(7)
_XB = _RNG.randn(_NB, _DIM).astype("float32")
_XQ = _XB[:8]


def _maxrss_bytes() -> int:
    # ru_maxrss is bytes on macOS, kibibytes on Linux; we only compare ratios
    # within one process, so the unit cancels out.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _one_cycle() -> None:
    db = elips.connect(":memory:", dimension=_DIM, device="metal")
    try:
        assert db.native.gpu_info().backend == "metal"
        _, labels = elips.bf_knn(_XQ, _XB, 5, "l2", "metal")
        # exact bf_knn: each query's nearest neighbour is itself.
        assert [int(labels[i][0]) for i in range(len(_XQ))] == list(range(len(_XQ)))
    finally:
        db.close()


def test_metal_create_destroy_cycles_are_bounded():
    warmup, measured = 20, 150

    for _ in range(warmup):  # pay one-time allocations / Metal JIT
        _one_cycle()
    before = _maxrss_bytes()

    for _ in range(measured):  # the loop under test
        _one_cycle()
    after = _maxrss_bytes()

    growth = after - before
    # ponytail: high-water-mark heuristic, not a precise leak detector. A
    # bounded workload plateaus after warmup; a per-cycle leak would scale the
    # growth with `measured`. 64 MiB over 150 small cycles is a generous ceiling
    # that still trips on a real unbounded leak. Tighten / swap for psutil RSS
    # sampling if false-negatives ever matter.
    assert growth < 64 * 1024 * 1024, (
        f"RSS grew {growth} bytes over {measured} Metal cycles — possible leak"
    )


def test_bf_knn_metal_repeated_dispatch_stays_correct():
    # Per-call GPU buffer alloc/free path, no engine object in the loop.
    _, first = elips.bf_knn(_XQ, _XB, 5, "l2", "metal")
    first = np.asarray(first)
    for _ in range(200):
        _, labels = elips.bf_knn(_XQ, _XB, 5, "l2", "metal")
        assert np.array_equal(np.asarray(labels), first)
