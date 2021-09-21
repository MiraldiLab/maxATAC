#! /usr/bin/env python3
import os
from setuptools import setup, find_packages

VERSION = "0.1.3"
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
      install_requires=["tensorflow==2.4.0",
                        "tensorboard",
                        "biopython",
                        "py2bit",
                        "pyBigWig==0.3.18",
                        "pydot",
                        "matplotlib",
                        "scikit-learn",
                        "pybedtools",
                        "pandas",
                        "pyfiglet",
                        "pyyaml",
                        "seaborn"],
      zip_safe=False,
      scripts=["maxatac/bin/maxatac"],
      classifiers=[]
      )
