import py2bit

from maxatac.utilities.helpers import get_absolute_path


def load_2bit(location):
    return py2bit.open(get_absolute_path(location))
