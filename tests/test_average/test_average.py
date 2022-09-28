import os
import pytest
from maxatac.analyses.average import run_averaging
from maxatac.utilities.system_tools import Namespace
from maxatac.utilities.constants import ALL_CHRS
import subprocess

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))

input_file_paths = ["../data/average/SRX10474876_Tn5_slop20_blacklisted.bw",
                    "../data/average/SRX10474877_Tn5_slop20_blacklisted.bw"]

expected_outputs = {"../data/average/IMR-90_single_chrom.bw": "6194b2bc8ddd1f8b9e277e21a79d9297",
                    "../data/average/IMR-90_multi_chrom.bw": "46775111a72df9fc54c8c9d153135272",
                    "../data/average/IMR-90_all_chrom.bw": "a7b2b7b1be22a046d49cff36ba46e8ea"}

chrom_sizes_path = "~/opt/maxatac/data/hg38/hg38.chrom.sizes"

def test_average_multi_chrom():
    """
    Test that maxatac average can produce a .bw that has been filtered for chr22 and chr3

    """
    args_test_multi_chrom = Namespace(bigwig_files=input_file_paths,
                                      output_dir=DATA_FOLDER,
                                      name="average_multi_chrom_test",
                                      chromosomes=["chr22", "chr3"],
                                      chromosome_sizes=chrom_sizes_path
                                      )

    run_averaging(args_test_multi_chrom)

    output_file = os.path.join(DATA_FOLDER, "average_multi_chrom_test.bw")

    results = subprocess.run(f"md5sum {output_file}", shell=True, capture_output=True)

    md5sum = str(results.stdout.decode()).split(" ")[0]

    assert md5sum == expected_outputs["../data/average/IMR-90_multi_chrom.bw"]


def test_average_single_chrom():
    """
    Test that maxatac average can produce a .bw file that has only a single chromosome

    """
    args_test_single_chrom = Namespace(bigwig_files=input_file_paths,
                                       output_dir=DATA_FOLDER,
                                       name="average_single_chrom_test",
                                       chromosomes=["chr22"],
                                       chromosome_sizes=chrom_sizes_path
                                       )

    run_averaging(args_test_single_chrom)

    output_file = os.path.join(DATA_FOLDER, "average_single_chrom_test.bw")

    results = subprocess.run(f"md5sum {output_file}", shell=True, capture_output=True)

    md5sum = str(results.stdout.decode()).split(" ")[0]

    assert md5sum == expected_outputs["../data/average/IMR-90_single_chrom.bw"]


def test_average_all_chrom():
    """
    Test that maxatac average can produce a .bw file that uses all chromosome

    """
    args_test_all_chrom = Namespace(bigwig_files=input_file_paths,
                                    output_dir=DATA_FOLDER,
                                    name="average_all_chrom_test",
                                    chromosomes=ALL_CHRS,
                                    chromosome_sizes=chrom_sizes_path
                                    )

    run_averaging(args_test_all_chrom)

    output_file = os.path.join(DATA_FOLDER, "average_all_chrom_test.bw")

    results = subprocess.run(f"md5sum {output_file}", shell=True, capture_output=True)

    md5sum = str(results.stdout.decode()).split(" ")[0]

    assert md5sum == expected_outputs["../data/average/IMR-90_all_chrom.bw"]
