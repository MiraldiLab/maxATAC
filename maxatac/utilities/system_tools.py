import logging
import pkg_resources
import os
import sys
import subprocess
from os import path, getcwd, makedirs, error, walk, environ
from maxatac.utilities.constants import CPP_LOG_LEVEL
from re import match
from multiprocessing import cpu_count


def get_absolute_path(p, cwd_abs_path=None):
    cwd_abs_path = getcwd() if cwd_abs_path is None else cwd_abs_path
    return p if path.isabs(p) else path.normpath(path.join(cwd_abs_path, p))


def get_cpu_count(reserved=0.25):  # reserved is [0, 1)
    avail_cpus = round((1 - reserved) * cpu_count())
    return 1 if avail_cpus == 0 else avail_cpus


def get_dir(dir: str, permissions=0o0775, exist_ok : bool=True):
    """Makes a directory at the given location

    Args:
        dir (str): Path of the directory
        permissions ([type], optional): Permissions of directory. Defaults to 0o0775.
        exist_ok (bool, optional): If True, program will continue if directory exists. Defaults to True.

    Returns:
        str: Absolute path to the created directory
        
    Example:
    
    >>> output_dir = get_dir("./output/")
    """
    abs_dir = get_absolute_path(dir)
    try:
        makedirs(abs_dir, mode=permissions)
    except error:
        if not exist_ok:
            raise
    return abs_dir


def get_files(current_dir, filename_pattern=None):
    filename_pattern = ".*" if filename_pattern is None else filename_pattern
    files_dict = {}
    for root, dirs, files in walk(current_dir):
        files_dict.update(
            {filename: path.join(root, filename) for filename in files if match(filename_pattern, filename)}
        )
    return files_dict


def get_rootname(l):
    return path.splitext(path.basename(l))[0]


def get_version():
    pkg = pkg_resources.require("maxatac")
    return pkg[0].version if pkg else "unknown version"


def remove_tags(l, tags):
    tags = tags if type(tags) is list else [tags]
    for tag in tags:
        l = l.replace(tag, "")
    return l


def replace_extension(l, ext):
    return get_absolute_path(
        path.join(
            path.dirname(l),
            get_rootname(l) + ext
        )
    )


def setup_logger(log_level, log_format):
    for log_handler in logging.root.handlers:
        logging.root.removeHandler(log_handler)

    for log_filter in logging.root.filters:
        logging.root.removeFilter(log_filter)

    logging.basicConfig(level=log_level, format=log_format)

    environ["TF_CPP_MIN_LOG_LEVEL"] = str(CPP_LOG_LEVEL[log_level])


class EmptyStream():

    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        pass


class Mute():
    NULL_FDS = []
    BACKUP_FDS = []

    def __enter__(self):
        self.suppress_stdout()

    def __exit__(self, type, value, traceback):
        self.restore_stdout()

    def suppress_stdout(self):
        self.NULL_FDS = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.BACKUP_FDS = os.dup(1), os.dup(2)
        os.dup2(self.NULL_FDS[0], 1)
        os.dup2(self.NULL_FDS[1], 2)

    def restore_stdout(self):
        os.dup2(self.BACKUP_FDS[0], 1)
        os.dup2(self.BACKUP_FDS[1], 2)
        os.close(self.NULL_FDS[0])
        os.close(self.NULL_FDS[1])


def check_data_packages_installed():
    """Check that packages are installed
    This module requires 
    """
    try:
        subprocess.run(["which", "git"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "git", "conda install git")
        
    try:
        subprocess.run(["which", "wget"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "wget", "conda install wget")


def raise_exception(e, package, install_link):
    print("command '{}' return with error (code {}): {}. Make sure {} is installed in your path {}.".format(e.cmd, e.returncode, e.output, package, install_link))
    sys.exit()


def check_prepare_packages_installed():
    try:
        subprocess.run(["which", "samtools"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "samtools", "http://www.htslib.org/")
        
    try:
        subprocess.run(["which", "bedtools"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "bedtools", "https://bedtools.readthedocs.io/")
        
    try:
        subprocess.run(["which", "pigz"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "pigz", "https://zlib.net/pigz/")
        
    try:
        subprocess.run(["which", "bedGraphToBigWig"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "bedGraphToBigWig", "https://anaconda.org/bioconda/ucsc-bedgraphtobigwig")
        