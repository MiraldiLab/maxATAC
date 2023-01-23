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
input_bam_path = "/Users/caz3so/workspaces/miraldiLab/maxATAC/tests/data/prepare/bulk_atac_SRX2717911_sub01.bam"

expected_bulk_outputs = {"GM12878_bulk_IS_slop20_RP20M.bw": "c49a30be1da68b73141ba02a2255a0f6",
                         "GM12878_bulk_IS_slop20_RP20M_minmax01.bw": "e0429a7fb990da36707a075343b97b3e",
                         "GM12878_bulk_IS_slop20_RP20M_minmax01_chromosome_min_max.txt": "36e5f37a4ec1aaea0fb55ed1c86e12ee",
                         "GM12878_bulk_IS_slop20_RP20M_minmax01_genome_stats.txt": "b366a2c759587d24440f25c6343caae6"}

# These are for the scATAC-seq processing functions
input_file_path = "../data/prepare/scatac_fragments.tsv"
expected_file_path = "../data/prepare/scatac_CutSites.tsv"
expected_dataframe = pd.read_table(expected_file_path, sep="\t")

# Full scATAC fragment file test
input_scatac_frags = "/Users/caz3so/workspaces/miraldiLab/maxATAC/tests/data/prepare/GM12878_fragments_subsample_1M.tsv"

expected_scatac_outputs = {"GM12878_scatac_1M_IS_slop20_RP20M.bw": "5ad8c991d458e74f96024337652c93b3",
                           "GM12878_scatac_1M_IS_slop20_RP20M_minmax01.bw": "04b7edda09ec6303d80c121109bad90e",
                           "GM12878_scatac_1M_IS_slop20_RP20M_minmax01_chromosome_min_max.txt": "25863a4d1b83e7410dbaae0fa6090d59",
                           "GM12878_scatac_1M_IS_slop20_RP20M_minmax01_genome_stats.txt": "a7397bf4c5ef4dd4c79aaab35830ff4d"}


def test_fragments_to_tn5_bed():
    """
    Test that fragments are converted to cut sites correctly

    """
    assert convert_fragments_to_tn5_bed(fragments_tsv=input_file_path,
                                        chroms=["chr1"]).equals(expected_dataframe)


def test_fragments_to_tn5_bed_wrong_chrom():
    """
    Test that fragments can be filtered by chromosome

    """
    assert (convert_fragments_to_tn5_bed(fragments_tsv=input_file_path,
                                         chroms=["chr2"]).equals(expected_dataframe) == False)


def test_prepare_bulk():
    """
    Test that a bam file can be prepared for maxATAC
    """
    # Args for bulk
    args_bulk = Namespace(input=input_bam_path,
                          output_dir=DATA_FOLDER,
                          name="GM12878_bulk",
                          chromosomes=AUTOSOMAL_CHRS,
                          chrom_sizes=chrom_sizes_path,
                          slop=20,
                          rpm_factor=20000000,
                          blacklist_bed=blacklist_path,
                          blacklist=blacklist_bigwig_path,
                          threads=4,
                          skip_dedup=False,
                          clip=False,
                          max=False,
                          method="min-max",
                          max_percentile=99)

    run_prepare(args_bulk)

    for file, md5 in expected_bulk_outputs.items():
        output_file = os.path.join(DATA_FOLDER, file)

        results = subprocess.run(f"md5sum {output_file}", shell=True, capture_output=True)

        md5sum = str(results.stdout.decode()).split(" ")[0]

        assert md5sum == md5


def test_prepare_scatac():
    """
    Test that a tsv file can be prepared for maxATAC
    """
    # Args for scatac
    args_scatac = Namespace(input=input_scatac_frags,
                            output_dir=DATA_FOLDER,
                            name="GM12878_scatac_1M",
                            chromosomes=AUTOSOMAL_CHRS,
                            chrom_sizes=chrom_sizes_path,
                            slop=20,
                            rpm_factor=20000000,
                            blacklist_bed=blacklist_path,
                            blacklist=blacklist_bigwig_path,
                            threads=4,
                            skip_dedup=False,
                            clip=False,
                            max=False,
                            method="min-max",
                            max_percentile=99
                            )

    run_prepare(args_scatac)

    for file, md5 in expected_scatac_outputs.items():
        output_file = os.path.join(DATA_FOLDER, file)

        results = subprocess.run(f"md5sum {output_file}", shell=True, capture_output=True)

        md5sum = str(results.stdout.decode()).split(" ")[0]

        assert md5sum == md5
