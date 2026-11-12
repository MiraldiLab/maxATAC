#! /usr/bin/env python3
import os
import subprocess
from setuptools import setup, find_packages

GIT_VERSION_FILE = os.path.join("maxatac", "git_version")
HERE = os.path.abspath(os.path.dirname(__file__))

# Version of an untagged commit. A development version makes `maxatac data` install the latest data
DEV_VERSION = "2.0.0.dev0"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=HERE, text=True, stderr=subprocess.DEVNULL).strip()


def get_version():
    """
    Tries to get package version with following order:
    1. in a git checkout: the tag on the current commit without its leading "v" (v2.0.0b1 -> 2.0.0b1),
       or DEV_VERSION if the commit is not tagged
    2. outside git (installing from an sdist): the git_version file written when the sdist was built
    3. DEV_VERSION
    Tags must be PEP 440 versions: `maxatac data` installs the data repository tag of the same name.
    Updates/creates git_version file with the package version
    Returns package version
    """

    version = DEV_VERSION

    try:
        git("rev-parse", "--git-dir")
        in_git = True
    except Exception:
        in_git = False

    if in_git:
        try:
            tag = git("describe", "--tags", "--exact-match")
            version = tag[1:] if tag.startswith("v") else tag
        except Exception:
            pass

    else:
        try:
            with open(GIT_VERSION_FILE, "r") as input_stream:
                version = input_stream.read().strip() or DEV_VERSION
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
    with open(README, "r", encoding="utf-8") as f:
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
    python_requires=">=3.9",
    install_requires=[
        # Keras 2 API: tf.keras.utils.OrderedEnqueuer and Adam(weight_decay=) need >=2.11,
        # and both were dropped with Keras 3 in TF 2.16
        "tensorflow>=2.14,<2.16",
        "tensorboard",
        "biopython",
        "py2bit",
        "pyBigWig",
        "pydot",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "pybedtools",
        "pandas",
        "pyfiglet",
        "pyyaml",
        "pysam",
        "seaborn",
        "pytest",
        "tqdm",
        "ushuffle"
    ],
    extras_require={
        # Only the multinomialnll* losses need it
        "multinomial": ["tensorflow-probability>=0.22,<0.24"],
    },
    zip_safe=False,
    scripts=["maxatac/bin/maxatac"],
    classifiers=[]
)
