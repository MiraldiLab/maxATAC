import pkg_resources

from multiprocessing import cpu_count
from os import path, getcwd, makedirs, error, walk
from re import match


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
