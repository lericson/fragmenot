from setuptools import Extension, setup
from Cython.Build import cythonize

ext_cts = Extension('cts', sources=['src/cts.pyx'], libraries=['m'])

setup(
    name='experiment',
    version='0.1',
    package_dir={'': 'src'},
    ext_modules=cythonize([ext_cts])
)
