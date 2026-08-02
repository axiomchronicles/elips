r"""ELIPS low-level binding facade.

This module re-exports the compiled :mod:`elips._core` extension through a
regular Python module so the low-level runtime API stays easy to import
directly.

The re-export list used to be maintained by hand -- 85 names spelled out twice,
once in an import block and once in ``__all__``. Adding a single ``py::class_``
in C++ therefore meant editing four Python-side locations, and forgetting one
was invisible until a user hit it. The surface is now taken from the extension
itself, so it cannot drift.

Examples::

    >>> import elips.core as core
    >>> db = core.open(":memory:", dimension=2, metric="cosine")
    >>> docs = db.vault("documents")
    >>> docs.count()
    0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elips import _native

if TYPE_CHECKING:
    from elips._native import *  # noqa: F403

__version__ = _native.__version__
_has_gpu = _native.has_gpu

globals().update({name: getattr(_native, name) for name in _native.__all__})

__all__ = [name for name in _native.__all__ if name != "has_gpu"]
