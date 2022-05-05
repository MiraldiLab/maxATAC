#! /usr/bin/env python3
import os
import time
import subprocess
from setuptools import setup, find_packages

GIT_VERSION_FILE = os.path.join("maxatac", "git_version")
HERE = os.path.abspath(os.path.dirname(__file__))


def get_git_tag():
    return subprocess.check_output(["git", "describe", "--contains"], text=True).split("^")[0].strip()


def get_git_timestamp():
    gitinfo = subprocess.check_output(
        ["git", "log", "--first-parent", "--max-count=1", "--format=format:%ct", "."], text=True).strip()
    return time.strftime("%Y%m%d%H%M%S", time.gmtime(int(gitinfo)))


def get_version():
    """
    Tries to get package version with following order:
    0. default version
    1. from git_version file - when installing from pip, this is the only source to get version
    2. from tag
    3. from commit timestamp
    Updates/creates git_version file with the package version
    Returns package version
    """

    version = "1.0.4"                                       # default version
    try:
        with open(GIT_VERSION_FILE, "r") as input_stream:   # try to get version info from file
            version = input_stream.read()
    except Exception:
        pass

    try:
        version = get_git_tag()                             # try to get version info from the closest tag
    except Exception:
        try:
            version = "1.0.4"          # try to get version info from commit date
        except Exception:
            pass

    try:
        with open(GIT_VERSION_FILE, "w") as output_stream:  # save updated version to file (or the same)
            output_stream.write(version)
    except Exception:
        pass
    return version


def get_description():
    README = os.path.join(HERE, "README.md")
    with open(README, "r") as f:
        return f.read()

setup(
    name="maxatac",
    description="maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    version=get_version(),
    url="https://github.com/MiraldiLab/maxATAC",
    download_url="https://github.com/MiraldiLab/maxATAC",
    author="Tareian, Faiz",
    author_email="tacazares@gmail.com, faizrizvi1993@gmail.com",
    license="Apache-2.0",
    include_package_data=True,
    packages=find_packages(
        exclude=[
            "data",
            "docs",
            "env",
            "packaging",
            "scripts",
            "tests"
        ]
    ),
    install_requires=[
        "tensorflow",
        "tensorboard",
        "biopython",
        "py2bit",
        "pyBigWig",
        "pydot",
        "matplotlib",
        "scikit-learn",
        "pybedtools",
        "pandas",
        "pyfiglet",
        "pyyaml",
        "pysam",
        "seaborn"
    ],
    zip_safe=False,
    scripts=["maxatac/bin/maxatac"],
    classifiers=[]
)
