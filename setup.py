#!/usr/bin/env python

import os
from setuptools import setup


def source_root_dir():
    """Return the path to the root of the source distribution"""
    return os.path.abspath(os.path.dirname(__file__))


def read_version():
    """Read the version from the ``lens.version`` module"""
    filename = os.path.join(source_root_dir(), "lens/version.py")
    with open(filename) as fin:
        namespace = {}
        exec(fin.read(), namespace)  # pylint: disable=exec-used
        return namespace["__version__"]


with open("README.rst") as file:
    LONG_DESCRIPTION = file.read()

setup(
    name="lens",
    version=read_version(),
    description="Summarise and explore Pandas DataFrames",
    copyright="Copyright 2017, ASI Data Science",
    license="Apache 2.0",
    url="https://github.com/asidatascience/lens",
    author="ASI Data Science",
    author_email="engineering@asidatascience.com",
    packages=["lens"],
    zip_safe=False,
    long_description=LONG_DESCRIPTION,
    install_requires=[
        "dask[dataframe,delayed]>=0.18.0",
        "ipywidgets>=6.0.0",
        "matplotlib",
        "numpy>=1.11",
        "pandas",
        "plotly>=3.0.0",
        "scipy",
        "tdigest>=0.5.0",
        "seaborn",
    ],
)
