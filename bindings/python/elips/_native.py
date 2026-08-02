r"""The single seam between the Python package and the compiled extension.

Every module in :mod:`elips` reaches the native runtime through this one
module. That is the whole point of it: the extension's symbol table used to be
mirrored by hand in ``elips/core.py``, so adding one ``py::class_`` in C++ meant
editing the binding, the re-export block, the ``__all__`` list, and the stub --
four mechanical edits, three of which were easy to forget. Anything imported
here is re-exported automatically, so that tax is gone.

This module is private. Public names are curated in the packages that build on
it; :mod:`elips.native` is the supported escape hatch for reaching native
handles directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elips import _core

if TYPE_CHECKING:
    # Re-exported for type checkers, which cannot see through the star import
    # below. At runtime these come from the extension via `globals().update`.
    from elips._core import *  # noqa: F403

__version__: str = _core.__version__

# True when the extension was compiled with GPU bindings. The GPU surface is
# conditional, so callers must not assume `GpuDevice` and friends exist.
has_gpu: bool = hasattr(_core, "GpuDevice")


def _public_names() -> tuple[str, ...]:
    """Every public symbol the extension exposes, in sorted order."""

    return tuple(sorted(name for name in dir(_core) if not name.startswith("_")))


# Bind the extension's public symbols into this module's namespace. Doing it
# programmatically is what keeps the seam from drifting: a new `py::class_`
# shows up here the moment it is compiled, with no Python-side edit at all.
globals().update({name: getattr(_core, name) for name in _public_names()})

__all__ = [*_public_names(), "has_gpu"]
