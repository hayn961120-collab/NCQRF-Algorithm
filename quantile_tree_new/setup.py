from pathlib import Path
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

ROOT = Path(__file__).parent

common_compile_args = ["-O3"]

extensions = [
    Extension(
        "qtrees.utils._typedefs",
        ["qtrees/utils/_typedefs.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
    ),
    Extension(
        "qtrees.utils._random",
        ["qtrees/utils/_random.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
    ),
    Extension(
        "qtrees.utils._utils",
        ["qtrees/utils/_utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
    ),
    Extension(
        "qtrees.utils._quantile",
        ["qtrees/utils/_quantile.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
    ),
    Extension(
        "qtrees.tree._criterion",
        ["qtrees/tree/_criterion.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
    ),
    Extension(
        "qtrees.tree._partitioner",
        ["qtrees/tree/_partitioner.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
    ),
    Extension(
        "qtrees.tree._splitter",
        ["qtrees/tree/_splitter.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
    ),
    Extension(
        "qtrees.tree._tree",
        ["qtrees/tree/_tree.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
        language="c++",
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={"boundscheck": False, "wraparound": False, "initializedcheck": False},
    ),
)
