#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To compile the cython code, run in your terminal:
    python setup.py build_ext --inplace
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import os
import platform


names = ['raytracing_ot']
sources = ['%s.pyx'%name for name in names]
extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ]

platform_name = platform.system()


if platform_name.lower() == 'darwin':
    versions = os.listdir('/usr/local/Cellar/gcc/')
    version = max(versions, key=lambda i: int(i.split('.')[0]))
    version_int = version.split('.')[0]
    path = '/usr/local/Cellar/gcc/%s/lib/gcc/%s'%(version, version_int)
    os.environ['CC'] = 'gcc-%s'%version_int
    os.environ['CXX'] = 'g++-%s'%version_int
    extra_link_args=['-Wl,-rpath,%s'%path]

else:
    extra_link_args=['-fopenmp']


for name, source in zip(names, sources):
    language = 'c++' if name!='_math' else None
    ext_modules=[Extension(name,
                           sources=[source],
                           extra_compile_args=extra_compile_args,
                           extra_link_args=extra_link_args,
                           language=language)
                 ]
        
    setup(name=name,
          cmdclass={"build_ext": build_ext},
          ext_modules=cythonize(ext_modules, language_level="3"),
          include_dirs=[numpy.get_include()]
    )


    






