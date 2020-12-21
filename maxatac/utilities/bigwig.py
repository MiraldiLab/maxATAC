import pyBigWig

from maxatac.utilities.helpers import get_absolute_path


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
