import os
import pytest
import pandas as pd
from maxatac.utilities.prepare_tools import convert_fragments_to_tn5_bed
from maxatac.analyses.prepare import run_prepare
from maxatac.utilities.constants import chrom_sizes_path, AUTOSOMAL_CHRS, blacklist_bigwig_path, blacklist_path

from maxatac.utilities.system_tools import Namespace
import subprocess

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))


def test_fragments_to_tn5_bed():
    """
    Test that fragments are converted to cut sites correctly

    """
    assert convert_fragments_to_tn5_bed(fragments_tsv=input_file_path,
                                        chroms=["chr1"]).equals(expected_dataframe)

