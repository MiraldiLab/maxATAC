import pytest
import os

from maxatac.utilities.system_tools import get_files


DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))


control_empty = []

control_all_files = [
    "hg19.2bit",
    "train_signal_cell_A549.bigwig",
    "validate_signal_cell_HCT116.bigwig",
    "validate_sites_cell_HCT116_tf_CTCF.bigwig",
    "average.bigwig",
    "predict_signal_cell_GM12878.bigwig",
    "train_sites_cell_A549_tf_CTCF.bigwig",
    "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
]

control_all_bigwigs = [
    "train_signal_cell_A549.bigwig",
    "validate_signal_cell_HCT116.bigwig",
    "validate_sites_cell_HCT116_tf_CTCF.bigwig",
    "average.bigwig",
    "predict_signal_cell_GM12878.bigwig",
    "train_sites_cell_A549_tf_CTCF.bigwig"
]

control_all_2bit = [
    "hg19.2bit"
]

control_files_by_names = [
    "hg19.2bit",
    "average.bigwig",
    "predict_signal_cell_GM12878.bigwig",
]

control_file_by_name = [
    "hg19.2bit"
]


@pytest.mark.parametrize(
    "filename_pattern, control_filenames",
    [
        (None, control_all_files),
        (".*bigwig", control_all_bigwigs),
        (".*2bit", control_all_2bit),
        ("hg19.2bit|average.bigwig|predict_signal_cell_GM12878.bigwig", control_files_by_names),
        ("hg19.2bit", control_file_by_name),
        ("mm10", control_empty)
    ]
)
def test_get_files(filename_pattern, control_filenames):
    collected_filenames = get_files(DATA_FOLDER, filename_pattern).keys()
    assert sorted(collected_filenames) == sorted(control_filenames)
