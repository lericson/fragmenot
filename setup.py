#!/usr/bin/env python

from setuptools import setup, find_packages
from Cython.Build import cythonize

ext_modules = cythonize('pyembree/*.pyx', language='c++')
for ext in ext_modules:
    ext.libraries = ['embree3']

setup(
    name='pyembree',
    version='0.2.0',
    ext_modules=ext_modules,
    zip_safe=False,
    setup_requires=['cython'],
    packages=find_packages(),
    package_data = {'pyembree': ['*.pxd']},
)
