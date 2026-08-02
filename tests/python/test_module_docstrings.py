"""Every ``elips`` module must expose a live ``__doc__``.

Placing ``from __future__ import annotations`` above the module docstring turns
the docstring into a discarded expression statement, which silently removes it
from ``help()`` and from ``pytest --doctest-modules`` collection.
"""

from __future__ import annotations

import importlib
import pkgutil

import elips


def _module_names() -> list[str]:
    return [
        info.name
        for info in pkgutil.walk_packages(elips.__path__, prefix="elips.")
        if not info.name.endswith("._core")
    ] + ["elips"]


def test_every_module_has_a_live_docstring() -> None:
    dead = [
        name
        for name in _module_names()
        if not (importlib.import_module(name).__doc__ or "").strip()
    ]
    assert not dead, f"modules with a lost or missing __doc__: {dead}"
