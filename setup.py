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
    description="maxATAC - Neural network for TF binding prediction",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    version=VERSION,
    url="https://github.com/MiraldiLab/maxATAC/tree/main",
    download_url="https://github.com/MiraldiLab/maxATAC.git",
    author="Miraldi Lab",
    author_email="",
    license="Apache-2.0",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "tensorflow-gpu==1.14.0",
        "tensorboard==1.14.0",
        "keras==2.2.5",
        "numpy==1.19.4",
        "pyBigWig==0.3.17",
        "py2bit==0.3.0",
        "pydot==1.4.1",
        "matplotlib",
        "scikit-learn",
        "pysam==0.15.3",
        "pandas==1.1.5",
        "h5py<3.0.0",
        "pybedtools==0.8.1",
        "pyfiglet",
        "tqdm"
    ],
    zip_safe=False,
    scripts=["maxatac/bin/maxatac"],
    classifiers=[]
)
