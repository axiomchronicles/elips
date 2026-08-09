"""Source build for the ELIPS Python package.

For local development the extension is built via the top-level CMake project
(`-DELIPS_BUILD_PYTHON=ON`), which places `_core` inside the `elips/` package.
This setup.py drives `cibuildwheel` / `pip` builds via CMake.

The C++ sources live at the repository root, outside this directory, so an
sdist has to carry its own copy of them under `core_src/`. That copy is made by
the `sdist` command below -- at build time, not at import time. Populating it
from module scope meant that merely importing this file (which `pip`, `build`,
and most editors do routinely) deleted and rewrote a directory tree, and left
behind an untracked duplicate of the engine that silently went stale.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
VENDORED_DIR = CURRENT_DIR / "core_src"

# Everything CMake needs to configure and build `elips_pymodule`. The CLI,
# benchmarks, and test suites are excluded deliberately: this build always
# passes -DELIPS_BUILD_{CLI,BENCH,TESTS}=OFF, so CMake never descends into
# them, and shipping them would only enlarge the sdist.
VENDORED_PATHS = ("CMakeLists.txt", "include", "src")


def _is_development_checkout(root: Path) -> bool:
    """True when `root` is the ELIPS repository rather than an unpacked sdist."""

    return (root / "CMakeLists.txt").exists() and (root / "src").exists()


def _cmake_source_dir() -> Path:
    """Return the tree to point CMake at.

    In a development checkout this is the repository itself, so a build always
    compiles the sources actually on disk. From an unpacked sdist it is the
    vendored `core_src/` copy.
    """

    if _is_development_checkout(REPO_ROOT):
        return REPO_ROOT
    if _is_development_checkout(VENDORED_DIR):
        return VENDORED_DIR
    raise SystemExit(
        "cannot locate the ELIPS C++ sources: expected a repository checkout "
        f"at {REPO_ROOT} or a vendored copy at {VENDORED_DIR}"
    )


def _vendor_sources() -> None:
    """Refresh `core_src/` with the sources an sdist needs to compile."""

    if VENDORED_DIR.exists():
        shutil.rmtree(VENDORED_DIR)
    VENDORED_DIR.mkdir(parents=True)

    for name in VENDORED_PATHS:
        source = REPO_ROOT / name
        if not source.exists():
            raise SystemExit(f"cannot build an sdist: {source} is missing")
        if source.is_dir():
            shutil.copytree(source, VENDORED_DIR / name)
        else:
            shutil.copy2(source, VENDORED_DIR / name)

    # CMake refers to the binding by its in-repo path, so mirror that layout.
    binding = VENDORED_DIR / "bindings" / "python"
    binding.mkdir(parents=True)
    shutil.copy2(CURRENT_DIR / "elips_python.cpp", binding / "elips_python.cpp")


class VendorSourcesSdist(sdist):
    """Vendor the C++ sources into `core_src/` before the archive is built."""

    def run(self) -> None:
        if _is_development_checkout(REPO_ROOT):
            _vendor_sources()
        super().run()


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        root = _cmake_source_dir()
        out_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = "Release"

        cmake_executable = "cmake"
        try:
            import cmake

            cmake_bin_dir = Path(cmake.CMAKE_BIN_DIR)
            if (cmake_bin_dir / "cmake").exists():
                cmake_executable = str(cmake_bin_dir / "cmake")
            elif (cmake_bin_dir / "cmake.exe").exists():
                cmake_executable = str(cmake_bin_dir / "cmake.exe")
        except (ImportError, AttributeError):
            pass

        cmake_args = [
            cmake_executable,
            "-S",
            str(root),
            "-B",
            self.build_temp,
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DELIPS_BUILD_PYTHON=ON",
            "-DELIPS_BUILD_TESTS=OFF",
            "-DELIPS_BUILD_CLI=OFF",
            "-DELIPS_BUILD_BENCH=OFF",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DELIPS_PYTHON_OUTPUT_DIR={out_dir}",
        ]
        local_pybind11 = root / "build" / "_deps" / "pybind11-src"
        if local_pybind11.exists():
            cmake_args.append(f"-DELIPS_PYBIND11_SOURCE_DIR={local_pybind11.resolve()}")

        if sys.platform == "darwin":
            archflags = os.environ.get("ARCHFLAGS", "")
            if "-arch x86_64" in archflags:
                cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=x86_64")
            elif "-arch arm64" in archflags:
                cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=arm64")
            elif "CMAKE_OSX_ARCHITECTURES" in os.environ:
                cmake_args.append(
                    f"-DCMAKE_OSX_ARCHITECTURES={os.environ['CMAKE_OSX_ARCHITECTURES']}"
                )
        subprocess.run(
            cmake_args,
            check=True,
        )
        subprocess.run(
            [
                cmake_executable,
                "--build",
                self.build_temp,
                "--config",
                cfg,
                "--target",
                "elips_pymodule",
                "-j",
            ],
            check=True,
        )


setup(
    ext_modules=[CMakeExtension("elips._core")],
    cmdclass={"build_ext": CMakeBuild, "sdist": VendorSourcesSdist},
    zip_safe=False,
)
