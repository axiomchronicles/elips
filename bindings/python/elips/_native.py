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


# Engine internals that used to sit in the public namespace, mapped to the
# module that owns them now. `None` marks a type with no public replacement: it
# is reachable here because the extension defines it, not because callers are
# meant to reach for it.
#
# The EQL AST moved rather than disappeared -- a parse still returns these
# types, so `elips.query` owns them. The index snapshots are different: nothing
# in the binding returns one, so they were only ever constructible placeholders.
DEMOTED: dict[str, str | None] = {
    "DeleteStatement": "elips.query",
    "FetchStatement": "elips.query",
    "IndexSnapshot": None,
    "IndexSnapshotKind": None,
    "InsertStatement": "elips.query",
    "IvfSnapshot": None,
    "PqSnapshot": None,
    "ScanStatement": "elips.query",
    "SearchStatement": "elips.query",
    "Token": "elips.query",
    "TokenKind": "elips.query",
    "VectorRef": "elips.query",
}


def supported_names() -> tuple[str, ...]:
    """The native surface a caller is meant to reach for.

    The seam itself stays complete -- it mirrors the extension, that is its job.
    This is the curated subset :mod:`elips.native` advertises.
    """

    return tuple(name for name in _public_names() if name not in DEMOTED)


def demotion_message(name: str, *, via: str) -> str:
    """Explain where a demoted name went, or that it never had a public home."""

    home = DEMOTED[name]
    if home is None:
        return (
            f"{via}.{name} is deprecated: it is an engine internal with no "
            f"public replacement, and nothing in the API returns one"
        )
    return f"{via}.{name} is deprecated; import {name} from {home} instead"


# Bind the extension's public symbols into this module's namespace. Doing it
# programmatically is what keeps the seam from drifting: a new `py::class_`
# shows up here the moment it is compiled, with no Python-side edit at all.
globals().update({name: getattr(_core, name) for name in _public_names()})

__all__ = [*_public_names(), "has_gpu"]
