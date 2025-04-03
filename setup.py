from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np


extensions = [
    Extension(
        "src.titan.low.decision_tree",
        ["src/titan/low/decision_tree.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions),
)