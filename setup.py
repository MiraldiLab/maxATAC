#! /usr/bin/env python3
import os
from setuptools import setup, find_packages

VERSION = "0.1.2"
HERE = os.path.abspath(os.path.dirname(__file__))


def get_description():
    README = os.path.join(HERE, 'README.md')
    with open(README, 'r') as f:
        return f.read()


setup(name="maxatac",
      description="Neural networks for predicting TF binding using ATAC-seq",
      long_description=get_description(),
      long_description_content_type="text/markdown",
      version=VERSION,
      url="",
      download_url="",
      author="",
      author_email="",
      license="Apache-2.0",
      include_package_data=True,
      packages=find_packages(),
      install_requires=["tensorflow-gpu==1.15.2",
                        "tensorboard",
                        "keras==2.3.1",
                        "py2bit==0.3.0",
                        "numpy==1.19.5",
                        "pyBigWig==0.3.17",
                        "pydot==1.4.1",
                        "matplotlib",
                        "scikit-learn==0.19",
                        "pysam==0.15.3",
                        "pybedtools==0.8.1",
                        "pandas==1.1.5",
                        "pyfiglet",
                        "h5py<3.0.0",
                        "grpcio==1.36.1",
                        "deeplift",
                        "seaborn",
                        "graphviz",
                        "shap @ git+https://github.com/AvantiShri/shap.git@master#egg=shap",
                        "modisco @ git+https://github.com/XiaotingChen/tfmodisco.git@0.5.9.2#egg-modisco"
                        ],
      zip_safe=False,
      scripts=["maxatac/bin/maxatac"],
      classifiers=[]
      )
