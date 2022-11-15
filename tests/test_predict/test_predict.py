import os
import pytest
import pandas as pd
from maxatac.utilities.prepare_tools import convert_fragments_to_tn5_bed
from maxatac.analyses.prepare import run_prepare
from maxatac.utilities.constants import chrom_sizes_path, AUTOSOMAL_CHRS, blacklist_bigwig_path, blacklist_path

from maxatac.utilities.system_tools import Namespace
import subprocess

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))

# These are for the bulk data sets
input_bw_path = "/Users/caz3so/workspaces/miraldiLab/maxATAC/tests/data/prepare/bulk_atac_SRX2717911_sub01.bam"

expected_outputs = {"GM12878_bulk_IS_slop20_RP20M.bw": "c49a30be1da68b73141ba02a2255a0f6",
                         "GM12878_bulk_IS_slop20_RP20M_minmax01.bw": "e0429a7fb990da36707a075343b97b3e",
                         "GM12878_bulk_IS_slop20_RP20M_minmax01_chromosome_min_max.txt": "36e5f37a4ec1aaea0fb55ed1c86e12ee",
                         "GM12878_bulk_IS_slop20_RP20M_minmax01_genome_stats.txt": "b366a2c759587d24440f25c6343caae6"}

# These are for the scATAC-seq processing functions
input_file_path = "../data/prepare/scatac_fragments.tsv"
expected_file_path = "../data/prepare/scatac_CutSites.tsv"


def test_fragments_to_tn5_bed():
    """
    Test that fragments are converted to cut sites correctly

    """
    assert convert_fragments_to_tn5_bed(fragments_tsv=input_file_path,
                                        chroms=["chr1"]
                                        ).equals(expected_dataframe)