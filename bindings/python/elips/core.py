r"""ELIPS low-level binding facade.

.. deprecated:: 1.1
    Use :mod:`elips.native` for the low-level runtime API. This module remains
    as a compatibility shim: it re-exports the extension's surface so existing
    imports keep working.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from elips import _native

if TYPE_CHECKING:
    from elips._native import *  # noqa: F403

warnings.warn(
    "elips.core is deprecated; import from elips.native instead",
    DeprecationWarning,
    stacklevel=2,
)

__version__ = _native.__version__
_has_gpu = _native.has_gpu

globals().update({name: getattr(_native, name) for name in _native.__all__})

__all__ = [name for name in _native.__all__ if name != "has_gpu"]
