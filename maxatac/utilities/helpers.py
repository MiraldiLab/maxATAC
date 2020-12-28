import pkg_resources
import pyBigWig
import py2bit
import os
import logging
import numpy as np
from multiprocessing import cpu_count
from os import path, getcwd, makedirs, error, walk, environ
from re import match
from maxatac.utilities.constants import CPP_LOG_LEVEL, DEFAULT_NORMALIZE_CHRS

### General Helpers ###
def get_absolute_path(p, cwd_abs_path=None):
    cwd_abs_path = getcwd() if cwd_abs_path is None else cwd_abs_path
    return p if path.isabs(p) else path.normpath(path.join(cwd_abs_path, p))


def get_version():
    pkg = pkg_resources.require("maxatac")
    return pkg[0].version if pkg else "unknown version"


def get_dir(dir, permissions=0o0775, exist_ok=True):
    abs_dir = get_absolute_path(dir)
    try:
        makedirs(abs_dir, mode=permissions)
    except error:
        if not exist_ok:
            raise
    return abs_dir


def get_rootname(l):
    return path.splitext(path.basename(l))[0]


def get_cpu_count(reserved=0.25):  # reserved is [0, 1)
    avail_cpus = round((1 - reserved) * cpu_count())
    return 1 if avail_cpus == 0 else avail_cpus


def replace_extension(l, ext):
    return get_absolute_path(
        path.join(
            path.dirname(l),
            get_rootname(l) + ext
        )
    )


def remove_tags(l, tags):
    tags = tags if type(tags) is list else [tags]
    for tag in tags:
        l = l.replace(tag, "")
    return l


def get_files(current_dir, filename_pattern=None):
    filename_pattern = ".*" if filename_pattern is None else filename_pattern
    files_dict = {}
    for root, dirs, files in walk(current_dir):
        files_dict.update(
            {filename: path.join(root, filename) for filename in files if match(filename_pattern, filename)}
        )
    return files_dict


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


def setup_logger(log_level, log_format):
    for log_handler in logging.root.handlers:
        logging.root.removeHandler(log_handler)
    
    for log_filter in logging.root.filters:
        logging.root.removeFilter(log_filter)
    
    logging.basicConfig(level=log_level, format=log_format)
    
    environ["TF_CPP_MIN_LOG_LEVEL"] = str(CPP_LOG_LEVEL[log_level])

##### Bigwigs Helpers #####

class EmptyStream():

    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        pass


def safe_load_bigwig(location):
    try:
        return pyBigWig.open(get_absolute_path(location))
    except (RuntimeError, TypeError):
        return EmptyStream()


def load_bigwig(location):
    return pyBigWig.open(get_absolute_path(location))


def dump_bigwig(location):
    return pyBigWig.open(get_absolute_path(location), "w")


def load_2bit(location):
    return py2bit.open(get_absolute_path(location))


def build_chrom_sizes_dict(genome_build):
    if genome_build == "hg19":
        #### hg19 chrom sizes
        num_bp=np.array([249250621,243199373,198022430,
                        191154276,180915260,171115067,
                        159138663,146364022,141213431,
                        135534747,135006516,133851895,
                        115169878,107349540,102531392,
                        90354753,81195210,78077248,
                        59128983,63025520,48129895,
                        51304566,155270560])

    else:
        #### hg38 chrom sizes
        num_bp=np.array([248956422,242193529,198295559,
                        190214555,181538259,170805979,
                        159345973,145138636,138394717,
                        133797422,135086622,133275309,
                        114364328,107043718,101991189,
                        90338345,83257441,80373285,
                        58617616,64444167,46709983,
                        50818468,156040895])

    # This will initialize an empty list to store the chrom sizes dict
    chromosome_length_dictionary={}

    # This will create the chromosome sizes dictionary
    for i in np.arange(len(DEFAULT_NORMALIZE_CHRS)):
        chromosome_length_dictionary[DEFAULT_NORMALIZE_CHRS[i]]=num_bp[i]

    return chromosome_length_dictionary
