import os
import pytest
from maxatac.analyses.average import run_averaging
from maxatac.utilities.system_tools import Namespace
from maxatac.utilities.constants import chrom_sizes_path
import subprocess

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))

input_file_paths = ["../data/average_data/SRX10474876_Tn5_slop20_blacklisted.bw",
                    "../data/average_data/SRX10474877_Tn5_slop20_blacklisted.bw"]

expected_outputs = {"../data/average_data/IMR-90_single_chrom.bw": "6194b2bc8ddd1f8b9e277e21a79d9297",
                    "../data/average_data/IMR-90_multi_chrom.bw": "46775111a72df9fc54c8c9d153135272"}

args_test_single_chrom = Namespace(bigwig_files=input_file_paths,
                                   output_dir=DATA_FOLDER,
                                   name="average_single_chrom_test",
                                   chromosomes=["chr22"],
                                   chromosome_sizes=chrom_sizes_path
                                   )

args_test_multi_chrom = Namespace(bigwig_files=input_file_paths,
                                  output_dir=DATA_FOLDER,
                                  name="average_multi_chrom_test",
                                  chromosomes=["chr22", "chr3"],
                                  chromosome_sizes=chrom_sizes_path
                                  )


def test_average_multi_chrom():
    """
    Test that maxatac average can produce a .bw that has been filtered for chr22 and chr3

    """
    run_averaging(args_test_multi_chrom)

    output_file = os.path.join(DATA_FOLDER, "average_multi_chrom_test.bw")

    results = subprocess.run(f"md5sum {output_file}", shell=True, capture_output=True)

    md5sum = str(results.stdout.decode()).split(" ")[0]

    assert md5sum == expected_outputs["../data/average_data/IMR-90_multi_chrom.bw"]


def test_average_single_chrom():
    """
    Test that maxatac average can produce a .bw file that has only a single chromosome

    """
    run_averaging(args_test_single_chrom)

    output_file = os.path.join(DATA_FOLDER, "average_single_chrom_test.bw")

    results = subprocess.run(f"md5sum {output_file}", shell=True, capture_output=True)

    md5sum = str(results.stdout.decode()).split(" ")[0]

    assert md5sum == expected_outputs["../data/average_data/IMR-90_single_chrom.bw"]
