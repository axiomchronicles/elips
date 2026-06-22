"""Source build for the ELIPS Python package.

For local development the extension is built via the top-level CMake project
(`-DELIPS_BUILD_PYTHON=ON`), which places `_core` inside the `elips/` package.
This setup.py drives `cibuildwheel` / `pip` builds via CMake.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Locate directory paths
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent

# If we are in the development repository, copy C++ sources to make bindings self-contained
if (REPO_ROOT / "CMakeLists.txt").exists() and (REPO_ROOT / "src").exists():
    core_src_dir = CURRENT_DIR / "core_src"
    if core_src_dir.exists():
        shutil.rmtree(core_src_dir)
    core_src_dir.mkdir(exist_ok=True)

    # Copy essential folders and files
    to_copy = ["src", "include", "cli", "benchmarks", "tests", "CMakeLists.txt"]
    for name in to_copy:
        src_path = REPO_ROOT / name
        dst_path = core_src_dir / name
        if src_path.exists():
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

    # Copy bindings/python/elips_python.cpp into core_src/bindings/python/elips_python.cpp
    bindings_py_dir = core_src_dir / "bindings" / "python"
    bindings_py_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CURRENT_DIR / "elips_python.cpp", bindings_py_dir / "elips_python.cpp")

    ROOT = core_src_dir
else:
    # We are running from a packaged sdist, the source files must already be in core_src
    ROOT = CURRENT_DIR / "core_src"
    if not ROOT.exists():
        # Fallback to parent workspace root just in case
        ROOT = CURRENT_DIR.parents[2]


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
            "-DELIPS_BUILD_BENCH=OFF",
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
    ext_modules=[CMakeExtension("elips._core")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)

