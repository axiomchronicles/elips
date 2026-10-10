#!/usr/bin/env python3
"""Reproducible datasets for the ELIPS ANN benchmark.

Two sources, both deterministic:

* :func:`random_dataset` — fixed-seed gaussian float32 (the default; needs no
  network, identical bytes on every machine).
* :func:`sift1m` — the standard SIFT1M corpus (1M x 128, ships ground truth)
  downloaded from the TEXMEX FTP/HTTP mirror and cached. Returns ``None``
  offline so callers degrade gracefully instead of crashing.

The SIFT1M path lets the benchmark cross-check recall/QPS against FAISS'
published numbers (methodology: docs/_redesign/_faiss_3H.md).
"""

from __future__ import annotations

import os
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

SEED = 1234
SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT_HTTP = "http://corpus-texmex.irisa.fr/sift.tar.gz"
DEFAULT_CACHE = os.path.expanduser("~/.cache/elips/datasets")


@dataclass
class Dataset:
    xb: np.ndarray  # (nb, d) float32 base vectors
    xq: np.ndarray  # (nq, d) float32 query vectors
    gt: np.ndarray | None  # (nq, k) int ground-truth neighbours, if shipped
    name: str


def random_dataset(dim: int, nb: int, nq: int, seed: int = SEED) -> Dataset:
    """Fixed-seed gaussian float32 base + query sets (no ground truth)."""
    rng = np.random.RandomState(seed)
    xb = rng.randn(nb, dim).astype("float32")
    xq = rng.randn(nq, dim).astype("float32")
    return Dataset(xb, xq, None, f"random(d={dim},nb={nb},nq={nq},seed={seed})")


# --------------------------------------------------------------------------- #
# SIFT1M (.fvecs / .ivecs) — little-endian: int32 dim, then `dim` values/row
# --------------------------------------------------------------------------- #
def read_vecs(path: str, dtype: str) -> np.ndarray:
    """Read a TEXMEX .fvecs/.ivecs file into an (n, dim) array."""
    raw = np.fromfile(path, dtype="int32")
    if raw.size == 0:
        return np.empty((0, 0), dtype=dtype)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)  # each row: [dim, v0..v_{dim-1}]
    assert np.all(rows[:, 0] == dim), f"ragged vecs file: {path}"
    return np.ascontiguousarray(rows[:, 1:]).view(dtype)


def _download(url: str, dest: str, timeout: float) -> bool:
    try:
        with (
            urllib.request.urlopen(url, timeout=timeout) as resp,
            open(dest, "wb") as fh,
        ):
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        if os.path.exists(dest):
            os.remove(dest)  # don't leave a partial file
        return False


def sift1m(
    cache: str = DEFAULT_CACHE, *, download: bool = True, timeout: float = 30.0
) -> Dataset | None:
    """Load SIFT1M from cache, downloading once if allowed. ``None`` if absent.

    Offline (or with ``download=False`` and no cache) returns ``None`` so the
    caller can fall back to :func:`random_dataset` instead of failing.
    """
    base = os.path.join(cache, "sift")
    needed = ("sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs")
    if not all(os.path.exists(os.path.join(base, f)) for f in needed):
        if not download:
            return None
        os.makedirs(cache, exist_ok=True)
        tgz = os.path.join(cache, "sift.tar.gz")
        if not (
            _download(SIFT_URL, tgz, timeout) or _download(SIFT_HTTP, tgz, timeout)
        ):
            return None  # offline: graceful skip
        with tarfile.open(tgz) as tf:
            tf.extractall(cache, filter="data")  # creates <cache>/sift/
        if not all(os.path.exists(os.path.join(base, f)) for f in needed):
            return None
    xb = read_vecs(os.path.join(base, "sift_base.fvecs"), "float32")
    xq = read_vecs(os.path.join(base, "sift_query.fvecs"), "float32")
    gt = read_vecs(os.path.join(base, "sift_groundtruth.ivecs"), "int32")
    return Dataset(xb, xq, gt.astype(np.int64), "SIFT1M")


def _selftest() -> None:
    import io
    import struct

    ds = random_dataset(8, 100, 5)
    assert ds.xb.shape == (100, 8) and ds.xb.dtype == np.float32
    assert np.array_equal(ds.xb, random_dataset(8, 100, 5).xb), "not reproducible"

    # round-trip a tiny .fvecs through read_vecs
    rows = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    buf = io.BytesIO()
    for r in rows:
        buf.write(struct.pack("<i", r.size))
        buf.write(r.tobytes())
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as fh:
        fh.write(buf.getvalue())
        name = fh.name
    try:
        got = read_vecs(name, "float32")
        assert np.array_equal(got, rows), got
    finally:
        os.remove(name)
    print("datasets selftest OK")


if __name__ == "__main__":
    _selftest()
