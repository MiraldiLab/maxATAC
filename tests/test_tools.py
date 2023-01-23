import pytest
import os
from maxatac.utilities.system_tools import get_files
from maxatac.utilities.constants import DATA_FOLDER

control_all_2bit = [
    "hg19.2bit"
]


@pytest.mark.parametrize(
    "filename_pattern, control_filenames",
    [
        ("hg38.2bit", control_all_2bit),
    ]
)

def test_get_one_hot_encoded(filename_pattern, control_filenames):
    collected_filenames = get_files(DATA_FOLDER, filename_pattern).keys()
    assert sorted(collected_filenames) == sorted(control_filenames)
