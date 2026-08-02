r"""The native runtime, exposed deliberately.

Everything the high-level API does goes through these objects, so reaching for
them is supported rather than a workaround. They live behind their own module
name so that dropping to the low-level API is visible at the call site: it is
``elips.native.Vault``, never a bare ``elips.Vault`` sitting next to
:class:`elips.Collection` with nothing to say which one a caller wanted.

Most code should not need this. Reach for it when you want an escape hatch the
typed layer does not cover -- :meth:`Vault.place_record` or a transaction.

Engine internals are not part of this surface. The EQL statement types, the
lexer's :class:`Token`, and the index snapshots used to be exported here; they
are still reachable for one minor version and warn, naming their new home.
:mod:`elips.query` owns the EQL AST now.

Examples::

    >>> import elips
    >>> db = elips.connect(":memory:", dimension=2)
    >>> vault = db.native.vault("documents")   # same object the collection wraps
    >>> _ = vault.place([1.0, 0.0], {"kind": "design"})
    >>> db.collection("documents").count()
    1
    >>> db.close()
"""

from __future__ import annotations

import warnings
from typing import Any

from elips import _native

__all__ = [*_native.supported_names(), "has_gpu"]


def __getattr__(name: str) -> Any:
    """Resolve a native symbol, warning for the ones that were demoted.

    Demoted names are deliberately absent from ``__all__`` but still resolve:
    removing them outright would break callers mid-version, so they get one
    minor version of a warning that names where the type lives now.
    """

    if name.startswith("_") or not hasattr(_native, name):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _native.DEMOTED:
        warnings.warn(
            _native.demotion_message(name, via=__name__),
            DeprecationWarning,
            stacklevel=2,
        )
        # Deliberately not cached: module `__getattr__` only runs when the name
        # is absent from `globals()`, so binding it here would silence every
        # later access and leave the deprecation visible at just one call site.
        return getattr(_native, name)
    value = getattr(_native, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
