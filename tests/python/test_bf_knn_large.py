"""Phase 8 §7.2 hardening: exact brute-force k-NN stays correct at larger DB.

Unit tests exercise `bf_knn` on tiny inputs; this is a smoke-sized but
*larger* database (thousands of vectors) checked against an independent numpy
exact L2 oracle. Because exact k-NN can tie, correctness is asserted on the
returned neighbours' true distances (not raw label equality), so ties never
make the test flaky. The Metal path, when present, must agree with the CPU
path to the same tolerance.
"""

from __future__ import annotations

import elips
import numpy as np
import pytest

_DIM = 32
_NB = 5000
_NQ = 40
_K = 10
_RNG = np.random.RandomState(2024)
_XB = _RNG.randn(_NB, _DIM).astype("float32")
_XQ = _RNG.randn(_NQ, _DIM).astype("float32")


def _true_sq_l2() -> np.ndarray:
    diff = _XQ[:, None, :] - _XB[None, :, :]
    return np.einsum("qbd,qbd->qb", diff, diff)  # (nq, nb) squared L2


def _assert_exact_topk(labels: np.ndarray, true: np.ndarray) -> None:
    labels = np.asarray(labels)
    assert labels.shape == (_NQ, _K)
    oracle = np.sort(true, axis=1)[:, :_K]
    for q in range(_NQ):
        got = np.sort(true[q, labels[q]])
        # atol absorbs float32 rounding between numpy and the kernel.
        assert np.allclose(got, oracle[q], atol=1e-2), (
            f"query {q}: {got} vs {oracle[q]}"
        )


def _has_metal() -> bool:
    return bool(getattr(elips, "_has_gpu", False)) and any(
        d.backend == "metal" for d in elips.gpu_devices()
    )


def test_bf_knn_cpu_matches_numpy_oracle():
    true = _true_sq_l2()
    _, labels = elips.bf_knn(_XQ, _XB, _K, "l2", "cpu")
    _assert_exact_topk(labels, true)


@pytest.mark.skipif(not _has_metal(), reason="no runtime Metal device")
def test_bf_knn_metal_matches_numpy_oracle_and_cpu():
    true = _true_sq_l2()
    _, cpu = elips.bf_knn(_XQ, _XB, _K, "l2", "cpu")
    _, metal = elips.bf_knn(_XQ, _XB, _K, "l2", "metal")
    _assert_exact_topk(metal, true)
    # CPU and Metal are both exact: their neighbours' distances must agree.
    cpu, metal = np.asarray(cpu), np.asarray(metal)
    for q in range(_NQ):
        assert np.allclose(
            np.sort(true[q, cpu[q]]), np.sort(true[q, metal[q]]), atol=1e-2
        )
