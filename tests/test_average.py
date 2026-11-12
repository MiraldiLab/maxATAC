import os

import pytest


def test_average_defaults_chrom_sizes_to_the_genome_bundle():
    # parse_arguments also runs update_reference_genome_paths, which fills args.chrom_sizes from --genome
    parser_module = pytest.importorskip("maxatac.utilities.parser")
    from maxatac.utilities.constants import maxatac_data_path

    args = parser_module.parse_arguments(["average", "-i", "a.bw", "b.bw", "-n", "average_test"])

    assert args.chrom_sizes == os.path.join(maxatac_data_path, "hg38/hg38.chrom.sizes")


def test_average_keeps_an_explicit_chrom_sizes_file():
    parser_module = pytest.importorskip("maxatac.utilities.parser")

    args = parser_module.parse_arguments(["average", "-i", "a.bw", "-n", "average_test",
                                          "--chromosome_sizes", "custom.chrom.sizes"])

    assert args.chrom_sizes == "custom.chrom.sizes"
