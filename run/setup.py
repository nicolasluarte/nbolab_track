from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('nbolab_track_cy.pyx'))
