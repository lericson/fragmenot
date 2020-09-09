from setuptools import Extension, setup
from Cython.Build import cythonize

ext_cts = Extension('cts', sources=['src/cts.pyx'], libraries=['m'])

setup(
    name='experiment',
    version='0.1',
    zip_safe=False,
    package_dir={'': 'src'},
    ext_modules=cythonize([ext_cts], build_dir='./var/build'),
    options={'build_ext': {'build_lib':  './var/build',
                           'build_temp': ''}}
)
