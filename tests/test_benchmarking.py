import os
import re
import hashlib

from maxatac.utilities.system_tools import get_files
from maxatac.utilities import parser

# Directory with all input files for testing.
# Subfolders are not allowed due to possible file name duplicates.
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))


CONTROL_MD5_SUMS = {
    '~/opt/maxatac/data/test/GM12878_CTCF_chr1_pred.bw': '69d59779a5edc3af8236413380a310cf',
    '~/opt/maxatac/data/test/GM12878_CTCF_chr1.bw': '69d59779a5edc3af8236413380a310cf',
    '~/opt/maxatac/data/hg38/hg38_maxatac_blacklist.bw': '436edd2edbf6b12282dc922c1bd5a64c'
}

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


print(args.blacklist)


def test_md5_sum():
    for filename, control_md5_sum in CONTROL_MD5_SUMS.items():
        md5_val = md5(os.path.expanduser(filename))
        assert md5_val == control_md5_sum, "change input file for tests"