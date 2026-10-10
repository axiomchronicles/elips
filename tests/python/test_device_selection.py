"""Phase 7.4: explicit device selection + no-silent-fallback (ground rule #9).

Covers the ``device=`` / ``fallback=`` surface on ``connect`` / ``Client`` and
the ``ELIPS_DEVICE`` environment override. A named GPU backend that is absent
must raise :class:`BackendUnavailableError`; ``fallback=True`` demotes to CPU
with a warning instead.
"""

from __future__ import annotations

import pytest

import elips


def _has_metal() -> bool:
    return bool(elips._has_gpu) and elips.GpuDeviceInfo().backend == "metal"


def _backend_of(engine) -> str:
    return engine.native.gpu_info().backend


def test_device_cpu_opens_on_cpu():
    db = elips.connect(":memory:", dimension=2, device="cpu")
    try:
        assert _backend_of(db) == "cpu"
    finally:
        db.close()


def test_client_device_cpu_opens_on_cpu():
    client = elips.Client(dimension=2, device="cpu")
    try:
        assert client.native.gpu_info().backend == "cpu"
    finally:
        client.close()


@pytest.mark.skipif(not _has_metal(), reason="no runtime Metal device")
def test_device_metal_uses_gpu():
    db = elips.connect(":memory:", dimension=2, device="metal")
    try:
        assert _backend_of(db) == "metal"
    finally:
        db.close()


def test_absent_backend_raises_backend_unavailable():
    # No CUDA on the Metal/CPU hosts this suite runs on.
    if elips._has_gpu and "cuda" in {
        d.backend for d in elips.gpu_devices()
    }:
        pytest.skip("a CUDA device is actually present")
    with pytest.raises(elips.BackendUnavailableError):
        elips.connect(":memory:", dimension=2, device="cuda:0")


def test_absent_backend_with_fallback_demotes_to_cpu_with_warning():
    if elips._has_gpu and "cuda" in {
        d.backend for d in elips.gpu_devices()
    }:
        pytest.skip("a CUDA device is actually present")
    with pytest.warns(UserWarning):
        db = elips.connect(
            ":memory:", dimension=2, device="cuda:0", fallback=True
        )
    try:
        assert _backend_of(db) == "cpu"
    finally:
        db.close()


def test_unknown_device_string_is_rejected():
    if not elips._has_gpu:
        pytest.skip("device-string parsing requires GPU bindings")
    with pytest.raises(ValueError):
        elips.connect(":memory:", dimension=2, device="banana")


def test_env_override_consulted_when_no_explicit_device(monkeypatch):
    monkeypatch.setenv("ELIPS_DEVICE", "cpu")
    db = elips.connect(":memory:", dimension=2)
    try:
        assert _backend_of(db) == "cpu"
    finally:
        db.close()


def test_explicit_device_beats_env_override(monkeypatch):
    monkeypatch.setenv("ELIPS_DEVICE", "cuda:0")
    # device="cpu" is explicit, so the (absent) CUDA env value is ignored.
    db = elips.connect(":memory:", dimension=2, device="cpu")
    try:
        assert _backend_of(db) == "cpu"
    finally:
        db.close()
