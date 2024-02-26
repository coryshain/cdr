#!/usr/bin/env python
from setuptools import setup
from setuptools import find_packages

setup(
    name='cdrnn',
    python_requires='>=3',
    version='0.8.5',
    description='A toolkit for continuous-time deconvolutional regression (CDR)',
    author='Cory Shain',
    author_email='cory.shain@gmail.com',
    url='https://github.com/coryshain/cdr',
    install_requires=[
        'dash',
        'numpy>=1.9.1',
        'pandas>=1.1',
        'pyyaml',
        'matplotlib',
        'tensorflow>=1.9',
        'tensorflow-probability',
        'scikit-learn',
        'scipy>=0.14'
    ],
    include_package_data=True,
    packages=find_packages(),
)
