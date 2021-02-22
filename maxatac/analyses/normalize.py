import logging
import pyBigWig
import os
import tqdm
from maxatac.utilities.system_tools import get_dir, Mute

with Mute():
    from maxatac.utilities.genome_tools import build_chrom_sizes_dict
    from maxatac.utilities.normalization_tools import find_genomic_min_max, minmax_normalize_array
    from maxatac.utilities.constants import DEFAULT_CHRS
    import numpy as np


def run_normalization(args):
    """
    Run minmax normalization on a bigwig file

    This function will min-max a bigwig file based on the minimum and maximum values in the chromosomes of interest.
    The code will loop through each chromosome and find the min and max values. It will then create a dataframe of
    the values per chromosome. It will then scale all other values between 0,1.
    _________________
    Workflow Overview

    1) Create directories and set up filenames
    2) Build a dictionary of the chromosomes sizes.
    3) Find the genomic min and max values by looping through each chromosome
    5) Loop through each chromosome and minmax normalize the values based on the genomic values.

    :param args: signal, output, chrom_sizes

    :return: A minmax normalized bigwig file
    """
    basename = os.path.basename(args.signal).split(".bw")[0]

    OUTPUT_FILENAME = os.path.join(args.output, basename + "_minmax01.bw")

    output_dir = get_dir(args.output)

    logging.error("Normalization" +
                  "\n  Input bigwig file: " + args.signal +
                  "\n  Output filename: " + OUTPUT_FILENAME +
                  "\n  Output directory: " + output_dir
                  )

    chromosome_length_dictionary = build_chrom_sizes_dict(args.chromosomes, args.chrom_sizes)

    genome_min, genome_max = find_genomic_min_max(args.signal)

    logging.error("Normalize and Write BigWig file")

    # TODO Parallelize this code for each chromosome
    with pyBigWig.open(args.signal) as input_bw, pyBigWig.open(OUTPUT_FILENAME, "w") as output_bw:
        header = [(x, chromosome_length_dictionary[x]) for x in sorted(DEFAULT_CHRS)]

        output_bw.addHeader(header)

        # Create a status bar for to look fancy and count what chromosome you are on
        chrom_status_bar = tqdm.tqdm(total=len(DEFAULT_CHRS), desc='Chromosomes Processed', position=0)

        for chrom_name, chrom_length in header:
            chr_vals = np.nan_to_num(input_bw.values(chrom_name, 0, chrom_length, numpy=True))

            normalized_signal = minmax_normalize_array(chr_vals, genome_min, genome_max)

            output_bw.addEntries(
                chroms=chrom_name,
                starts=0,
                ends=chrom_length,
                span=1,
                step=1,
                values=normalized_signal.tolist()
            )

            chrom_status_bar.update(1)

    logging.error("Results saved to: " + output_dir)
