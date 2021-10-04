import os
import re
import hashlib

from maxatac.utilities.system_tools import get_files

# Directory with all input files for testing.
# Subfolders are not allowed due to possible file name duplicates.
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))


CONTROL_MD5_SUMS = {
    "hg19.2bit": "bcdbfbe9da62f19bee88b74dabef8cd3",
    "train_signal_cell_A549.bigwig": "d41d8cd98f00b204e9800998ecf8427e",
    "validate_signal_cell_HCT116.bigwig": "d41d8cd98f00b204e9800998ecf8427e",
    "validate_sites_cell_HCT116_tf_CTCF.bigwig": "d41d8cd98f00b204e9800998ecf8427e",
    "average.bigwig": "d41d8cd98f00b204e9800998ecf8427e",
    "predict_signal_cell_GM12878.bigwig": "d41d8cd98f00b204e9800998ecf8427e",
    "train_sites_cell_A549_tf_CTCF.bigwig": "d41d8cd98f00b204e9800998ecf8427e",
    "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed": "63c2fda65d6810d4194e92c8ea8e394d"
}


def get_md5_sum(location, block_size=2**20):
    md5_sum = hashlib.md5()
    with open(location , "rb") as input_stream:
        while True:
            buf = input_stream.read(block_size)
            if not buf:
                break
            md5_sum.update(buf)
    return md5_sum.hexdigest()


def test_md5_sum():
    collected_md5_sums = {}
    for filename, location in get_files(DATA_FOLDER).items():
        collected_md5_sums[filename] = get_md5_sum(location)
    for filename, control_md5_sum in CONTROL_MD5_SUMS.items():
        assert filename in collected_md5_sums, "missing input file for tests"
        assert collected_md5_sums[filename] == control_md5_sum, "change input file for tests"
