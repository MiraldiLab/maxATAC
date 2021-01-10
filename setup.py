#! /usr/bin/env python3
import os
from setuptools import setup, find_packages


VERSION = "0.0.3"
HERE = os.path.abspath(os.path.dirname(__file__))


def get_description():
    README = os.path.join(HERE, 'README.md')
    with open(README, 'r') as f:
        return f.read()


setup(
    name="maxatac",
    description="maxATAC - Dilated convolutional neural network \
         for TF binding prediction from ATAC-seq",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    version=VERSION,
    url="https://github.com/MiraldiLab/maxATAC.git",
    download_url="https://github.com/MiraldiLab/maxATAC.git",
    author="Miraldi Lab",
    author_email="emily.miraldi@cchmc.org",
    license="Apache-2.0",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "tensorflow-gpu==1.14.0",
        "tensorboard==1.14.0",
        "keras==2.2.5",
        "pyBigWig==0.3.16",
        "py2bit==0.3.0",
        "numpy==1.19.4",
        "matplotlib",
        "scikit-learn",
        "pandas==1.1.5",
        "pysam==0.15.3",
        "pybedtools==0.8.1"
    ],
    zip_safe=False,
    scripts=["maxatac/bin/maxatac"],
    classifiers=[]
)
