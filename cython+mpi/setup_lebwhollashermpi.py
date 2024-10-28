# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy as np


compile_args=["/openmp"]

extensions = [
    Extension(
        "lebwhollashermpi",
        ["lebwhollashermpi.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    )
]

setup(
    name="lebwhollashermpi",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
