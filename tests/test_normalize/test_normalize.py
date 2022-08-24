import os
import pytest
from maxatac.analyses.normalize import run_normalization
from maxatac.utilities.system_tools import Namespace
import subprocess
from maxatac.utilities.constants import blacklist_bigwig_path, AUTOSOMAL_CHRS, chrom_sizes_path
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))

input_file_paths = "../data/average/IMR-90_all_chrom.bw"

expected_outputs = {"../data/normalize/IMR-90_minmax01_percentile99.bw": "413b925d425b039ff3ac46d84bd790c9",
                    "../data/normalize/IMR-90_minmax01_percentile99_genome_stats.txt": "cb54a14503617564674d1ba398a8e05e",
                    "../data/normalize/IMR-90_minmax01_percentile99_chromosome_min_max.txt": "19185afdd98c18ab03c891ed9f51f06c"}

args_test_min_max = Namespace(signal=input_file_paths,
                              output_dir=DATA_FOLDER,
                              name="normalize_test",
                              max=False,
                              method="min-max",
                              max_percentile=99,
                              blacklist=blacklist_bigwig_path,
                              chromosomes=AUTOSOMAL_CHRS,
                              chrom_sizes=chrom_sizes_path,
                              clip=False)


def test_normalization():
    """
    Test that maxatac normalize can min-max normalize a bigwig file to the 99th percentile max

    """
    run_normalization(args_test_min_max)

    output_file = os.path.join(DATA_FOLDER, "normalize_test.bw")

    results = subprocess.run(f"md5sum {output_file}", shell=True, capture_output=True)

    md5sum = str(results.stdout.decode()).split(" ")[0]

    assert md5sum == expected_outputs["../data/normalize/IMR-90_minmax01_percentile99.bw"]

