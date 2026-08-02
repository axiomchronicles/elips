"""The Python package must reach the extension through exactly one seam.

`elips/core.py` used to be a hand-maintained mirror of the extension's symbol
table: 85 names spelled out in an import block and again in `__all__`. Adding
one `py::class_` in C++ meant editing four Python-side locations, and missing
one was invisible until a user hit it.

These tests pin the two properties that keep that from coming back: the surface
is derived from the extension rather than written down, and every module goes
through `elips._native` instead of importing `elips._core` directly.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import elips
from elips import _core, _native

PACKAGE_DIR = pathlib.Path(elips.__file__).parent
SOURCE_FILES = sorted(
    path
    for path in PACKAGE_DIR.rglob("*.py")
    if "__pycache__" not in path.parts
)


def test_the_seam_exposes_every_public_extension_symbol() -> None:
    """`_native` must not be able to drift from the extension."""

    extension = {name for name in dir(_core) if not name.startswith("_")}
    seam = set(_native.__all__) - {"has_gpu"}
    assert seam == extension, (
        f"seam is missing {sorted(extension - seam)}, "
        f"invents {sorted(seam - extension)}"
    )


def test_core_facade_matches_the_seam() -> None:
    """`elips.core` re-exports the seam wholesale rather than a copied list."""

    import elips.core

    assert set(elips.core.__all__) == set(_native.__all__) - {"has_gpu"}


def test_no_module_hardcodes_a_reexport_list() -> None:
    """No module may spell out the extension's symbols by hand.

    A literal `__all__` is fine for a module that defines its own names. What
    is not fine is a literal list of *native* symbols, which is what drifted.
    """

    native_symbols = {name for name in dir(_core) if not name.startswith("_")}
    offenders: list[str] = []

    for path in SOURCE_FILES:
        if path.name in {"_native.py", "core.py"}:
            continue  # the seam and its facade, both generated
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                continue
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                continue
            listed = {
                el.value
                for el in node.value.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            }
            # A handful of overlapping names is ordinary re-export; dozens
            # means the symbol table was transcribed.
            overlap = listed & native_symbols
            if len(overlap) > 20:
                offenders.append(
                    f"{path.relative_to(PACKAGE_DIR)}: {len(overlap)} native "
                    f"symbols written out by hand"
                )

    assert not offenders, "; ".join(offenders)


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_modules_import_the_seam_not_the_extension(path: pathlib.Path) -> None:
    """Only `_native` may import `_core`."""

    if path.name == "_native.py":
        pytest.skip("the seam is the one module allowed to import the extension")

    tree = ast.parse(path.read_text())
    direct = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {"elips._core", "_core"}
    ] + [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        and any(alias.name.endswith("_core") for alias in node.names)
    ]

    assert not direct, (
        f"{path.name} imports the extension directly at line(s) {direct}; "
        f"import from elips._native instead"
    )


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_imports_are_absolute(path: pathlib.Path) -> None:
    """The project standard is absolute imports (audit §2)."""

    tree = ast.parse(path.read_text())
    relative = [
        f"line {node.lineno}: {'.' * node.level}{node.module or ''}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert not relative, f"{path.name} uses relative imports: {relative}"
