from setuptools import Extension, setup
from Cython.Build import cythonize

ext_tree_search = Extension('tree_search', sources=['src/tree_search.pyx'], libraries=['m'])

setup(
    name='experiment',
    version='0.1',
    zip_safe=False,
    package_dir={'': 'src'},
    ext_modules=cythonize([ext_tree_search], build_dir='./var/build', annotate=True),
    options={'build_ext': {'build_lib':  './var/build',
                           'build_temp': ''}}
)
