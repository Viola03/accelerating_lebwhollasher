# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy as np



# setup(
#     ext_modules=cythonize("lebwhollasher.pyx", compiler_directives={'language_level': "3"}),
#     include_dirs=[np.get_include()],
#     extra_compile_args=["-fopenmp"],
#     extra_link_args=["-fopenmp"],
# )

compile_args=["/openmp"]

extensions = [
    Extension(
        "lebwhollasher",
        ["lebwhollasher.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    )
]

setup(
    name="lebwhollasher",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)

