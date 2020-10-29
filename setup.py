#!/usr/bin/env python
from setuptools import setup
from setuptools import find_packages

setup(name='CDR',
      version='0.3.0',
      description='A toolkit for continuous-time deconvolutional regression (CDR)',
      author='Cory Shain',
      author_email='cory.shain@gmail.com',
      url='https://github.com/coryshain/cdr',
      dependency_links=['git+https://github.com/coryshain/edward@master#egg=edward-0'],
      install_requires=['numpy>=1.9.1',
                        'pandas>=0.19.2',
                        'matplotlib',
                        'tensorflow>=1.9,<2.0',
                        'scipy>=0.14'],
      packages=find_packages(),
)