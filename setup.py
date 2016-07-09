#!/usr/bin/env python

# python setup.py sdist bdist_wheel --universal

from __future__ import print_function
#from distutils.core import setup
from setuptools import setup

setup(name='npplus',
      version='0.9.4',
      description='Enhancements to Numpy',
      long_description=open('README.md').read(),
      author='David H. Munro',
      author_email='dhmunro@users.sourceforge.net',
      url='https://github.com/dhmunro/npplus',
      packages=['npplus', 'npplus/pyplotx'],
      install_requires=['numpy>=1.5.1', 'scipy>=0.9.0', 'matplotlib>=1.0.1'],
      license='http://opensource.org/licenses/BSD-2-Clause',
      platforms=['Linux', 'MacOS X', 'Unix'],
      classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Unix',
        ],
      #cmdclass = {'test': TestCommand},
      )
