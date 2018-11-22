from distutils.core import setup, Extension
from Cython.Build import cythonize


ihmm = Extension("", sources=['./ihmm/hdp.pyx','./ihmm/ihmm.pyx','./ihmm/samplers.pyx'])

setup(name="ihmm",
      ext_modules=cythonize(ihmm))
