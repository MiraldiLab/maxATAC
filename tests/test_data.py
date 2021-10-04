import os
import re
import hashlib

from maxatac.utilities.system_tools import get_files

# Directory with all input files for testing.
# Subfolders are not allowed due to possible file name duplicates.
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))


CONTROL_MD5_SUMS = {
    "hg19.2bit": "bcdbfbe9da62f19bee88b74dabef8cd3",
    "train_signal_cell_A549.bigwig": "ee39f3948066dff73174885ed9270777",
    "validate_signal_cell_HCT116.bigwig": "9227d8cad5944452feff5054e6f68dc1",
    "validate_sites_cell_HCT116_tf_CTCF.bigwig": "e73b320b1d3efd04e670cf8477f00021",
    "average.bigwig": "56274bc90f52f66ae7d4c3d2592c85bb",
    "predict_signal_cell_GM12878.bigwig": "7c445dd4737ecaa550cf3582b7fcb098",
    "train_sites_cell_A549_tf_CTCF.bigwig": "f4805b3224bd29bebaf23485446a2ab2",
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
