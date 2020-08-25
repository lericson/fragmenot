from setuptools import setup
from Cython.Build import cythonize

setup(
    name='experiment',
    version='0.1',
    package_dir={'': 'src'},
    ext_modules=cythonize('src/cts.pyx')
)
