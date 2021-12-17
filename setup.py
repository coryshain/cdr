#!/usr/bin/env python
from setuptools import setup
from setuptools import find_packages

setup(
    name='CDR',
    python_requires='>=3',
    version='0.5.1',
    description='A toolkit for continuous-time deconvolutional regression (CDR)',
    author='Cory Shain',
    author_email='cory.shain@gmail.com',
    url='https://github.com/coryshain/cdr',
    install_requires=[
        'dash',
        'numpy>=1.9.1',
        'pandas>=1.1',
        'matplotlib',
        'tensorflow>=1.9',
        'tensorflow-probability',
        'scikit-learn',
        'scipy>=0.14'
    ],
    package_data=[
        'cdr/templates/cdr_model_template.ini',
        'cdr/templates/cdr_plot_template.ini',
    ],
    include_package_data=True,
    packages=find_packages(),
)