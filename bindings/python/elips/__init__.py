from __future__ import annotations

r"""ELIPS — embedded, local-storage-first vector retrieval.

The package exposes two layers:

- :mod:`elips.core` for the low-level bindings that mirror the C++ runtime.
- :mod:`elips.modern` for the typed ergonomic :class:`Engine` /
  :class:`Arena` wrapper.

Examples::

    >>> import elips
    >>> engine = elips.connect(":memory:", dimension=2)
    >>> arena = engine.arena("documents")
    >>> _ = arena.write(text="alpha design note", meta={"kind": "design"})
    >>> arena.probe_text("alpha", top=1)[0].text
    'alpha design note'
    >>> engine.close()
"""

from . import core as _core_api
from . import modern as _modern_api
from .core import *  # noqa: F401,F403
from .modern import *  # noqa: F401,F403

__version__ = _core_api.__version__
_has_gpu = _core_api._has_gpu

__all__ = [*_core_api.__all__, *_modern_api.__all__]
