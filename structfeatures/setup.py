#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

PACKAGE_NAME = "structfeatures"
REPO_NAME = "structfeatures"


def read(*names, **kwargs):
    with io.open(
            join(dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    description='Degree and local clustering coefficient features',
    long_description="",
    long_description_content_type='text/markdown',
    author='Ciwan Ceylan',
    author_email='ciwan@kth.se',
    url='',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    keywords=[],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.15', 'scipy>=1.0.0', 'numba>=0.50.1',
    ],
    extras_require={}
)
