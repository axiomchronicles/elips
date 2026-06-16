"""Source build for the ELIPS Python package.

For local development the extension is built via the top-level CMake project
(`-DELIPS_BUILD_PYTHON=ON`), which places `_core` inside the `elips/` package.
This setup.py drives `cibuildwheel` / `pip` builds via CMake.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parents[2]


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        out_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = "Release"
        cmake_args = [
            "cmake", "-S", str(ROOT), "-B", self.build_temp,
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DELIPS_BUILD_PYTHON=ON",
            "-DELIPS_BUILD_TESTS=OFF",
            "-DELIPS_BUILD_CLI=OFF",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DELIPS_PYTHON_OUTPUT_DIR={out_dir}",
        ]
        local_pybind11 = ROOT / "build" / "_deps" / "pybind11-src"
        if local_pybind11.exists():
            cmake_args.append(
                f"-DELIPS_PYBIND11_SOURCE_DIR={local_pybind11.resolve()}"
            )
        subprocess.run(
            cmake_args,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", self.build_temp, "--target", "elips_pymodule",
             "-j"],
            check=True,
        )


setup(
    name="elips",
    version="1.0.0",
    description="Embedded local vector database (SQLite for vectors)",
    packages=find_packages(include=["elips", "elips.*"]),
    package_data={"elips": ["py.typed", "_core.pyi"]},
    ext_modules=[CMakeExtension("elips._core")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    zip_safe=False,
)
