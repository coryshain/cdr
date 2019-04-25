#!/usr/bin/env python
from setuptools import setup
from setuptools import find_packages

setup(name='DTSR',
      version='0.2.0',
      description='A toolkit for Deconvolutional Time Series Regression (DTSR)',
      author='Cory Shain',
      author_email='cory.shain@gmail.com',
      url='https://github.com/coryshain/dtsr',
      install_requires=['numpy>=1.9.1',
                        'pandas>=0.19.2',
                        'matplotlib',
                        'tensorflow==1.6.0',
                        'scipy>=0.14'],
      packages=find_packages(),
)