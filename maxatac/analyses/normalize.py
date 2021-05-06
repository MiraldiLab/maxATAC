import logging
import pyBigWig
import os
from maxatac.utilities.system_tools import get_dir, Mute
import sys

with Mute():
    from maxatac.utilities.genome_tools import build_chrom_sizes_dict
    from maxatac.utilities.normalization_tools import find_genomic_min_max, minmax_normalize_array
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
    # Set up the names and directories
    basename = os.path.basename(args.signal).split(".bw")[0]

    if args.log_transform:
        OUTPUT_FILENAME = os.path.join(args.output, basename + "_logp1_minmax01.bw")

    else:
        OUTPUT_FILENAME = os.path.join(args.output, basename + "_minmax01.bw")

    output_dir = get_dir(args.output)

    logging.error("Normalization" +
                  "\n  Input bigwig file: " + args.signal +
                  "\n  Output filename: " + OUTPUT_FILENAME +
                  "\n  Output directory: " + output_dir
                  )

    # Build a dictionary of chrom sizes to use to write the bigwig
    chromosome_length_dictionary = build_chrom_sizes_dict(args.chroms, args.chrom_sizes)

    if args.max:
        logging.error("Using provided minimum and maximum values for normalization")

        genome_max = args.max
        genome_min = args.min

        logging.error("Minimum value: " + str(args.min) + "\n"
                      "Maximum value: " + str(args.max))

    else:
        logging.error("Find genomic minimum and maximum values")

        # Find the genomic min and maximum values
        genome_min, genome_max = find_genomic_min_max(args.signal, chrom_sizes_dict=chromosome_length_dictionary)

        logging.error("Minimum value to use: " + str(args.min))
        logging.error("Maximum value to use: " + str(args.max))

    logging.error("Normalize and Write BigWig file")

    # TODO Parallelize this code for each chromosome
    with pyBigWig.open(args.signal) as input_bw, pyBigWig.open(OUTPUT_FILENAME, "w") as output_bw:
        header = [(x, chromosome_length_dictionary[x]) for x in sorted(args.chroms)]

        output_bw.addHeader(header)

        for chrom_name, chrom_length in header:
            chr_vals = np.nan_to_num(input_bw.values(chrom_name, 0, chrom_length, numpy=True))

            if args.log_transform:
                chr_vals = np.log(chr_vals + 1)

            normalized_signal = minmax_normalize_array(chr_vals, genome_min, genome_max)

            output_bw.addEntries(chroms=chrom_name,
                                 starts=0,
                                 ends=chrom_length,
                                 span=1,
                                 step=1,
                                 values=normalized_signal.tolist()
                                 )

        output_bw.close()
        input_bw.close()

    logging.error("Results saved to: " + output_dir)

    sys.exit()
