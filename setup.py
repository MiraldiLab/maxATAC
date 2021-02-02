#! /usr/bin/env python3
import os
from setuptools import setup, find_packages


VERSION = "0.0.1"
HERE = os.path.abspath(os.path.dirname(__file__))


def get_description():
    README = os.path.join(HERE, 'README.md')
    with open(README, 'r') as f:
        return f.read()


setup(
    name="maxatac",
    description="maxATAC - DeepCNN for motif binding prediction",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    version=VERSION,
    url="https://bitbucket.org/miraldilab/maxatac/src",
    download_url="https://bitbucket.org/miraldilab/maxatac/src",
    author="Michael Kotliar",
    author_email="misha.kotliar@gmail.com",
    license="Apache-2.0",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "tensorflow-gpu==1.14.0",
        "keras==2.2.5",
        "pyBigWig==0.3.16",
        "py2bit==0.3.0",
        "numpy==1.16.2",
        "pydot==1.4.1",
        "matplotlib==3.2.1",
        "scikit-learn==0.22.2",
        "pandas==1.0.3"
    ],
    zip_safe=False,
    scripts=["maxatac/bin/maxatac"],
    classifiers=[]
)
