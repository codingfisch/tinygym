import numpy
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_kwargs = {'extra_compile_args': ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION', '-O1']}
ext_mods = [Extension(f'tinygym.envs.{Path(fp.stem)}', sources=[fp], **ext_kwargs) for fp in Path('tinygym/envs').glob('*.pyx')]
setup(name='tinygym', ext_modules=cythonize(ext_mods), packages=['tinygym', 'tinygym.envs'],
      include_dirs=[numpy.get_include()])
