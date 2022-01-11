import logging

import numpy as np
import pyBigWig

from maxatac.utilities.system_tools import get_dir, Mute
import os

with Mute():
    from maxatac.utilities.genome_tools import build_chrom_sizes_dict, chromosome_blacklist_mask
    from maxatac.utilities.normalization_tools import minmax_normalize_array, \
        median_mad_normalize_array, zscore_normalize_array, get_genomic_stats, arcsinh_normalize_array


def run_normalization(args):
    """
    Normalize a bigwig file
    
    This function will normalize a bigwig array based on the desired method:

    -median-mad normalization
    -min-max normalization
    -zscore normalization

    The code will loop through each chromosome and find the min and max values. It will then create a dataframe of
    the values per chromosome. It will then scale all other values.
    _________________
    Workflow Overview

    1) Create directories and set up filenames
    2) Build a dictionary of the chromosomes sizes.
    3) Find the genomic min and max values by looping through each chromosome
    5) Loop through each chromosome and minmax normalize the values based on the genomic values.

    :param args: signal, output, chrom_sizes, chroms, max_percentile, method, blacklist

    :return: A minmax normalized bigwig file
    """
    OUTPUT_FILENAME = os.path.join(args.output, args.prefix + ".bw")

    # Set up the output directoryz
    output_dir = get_dir(args.output)

    logging.error("Normalization" +
                  "\n  Input bigwig file: " + args.signal +
                  "\n  Output filename: " + OUTPUT_FILENAME +
                  "\n  Output directory: " + output_dir +
                  "\n  Using " + args.method + " normalization"
                  )

    # Build a dictionary of chrom sizes to use to write the bigwig
    chromosome_length_dictionary = build_chrom_sizes_dict(args.chroms, args.chrom_sizes)

    # If the user provides a max to normalize to, use that value
    if args.max:
        logging.error("Using provided minimum and maximum values for normalization")
        logging.error("Minimum value: " + str(args.min) + "\n" + "Maximum value: " + str(args.max))
        max_value = args.max
        min_value = args.min
    
    else:
        logging.error("Calculating stats per chromosome")

        min_value, max_value, median, mad, mean_value, std_value = get_genomic_stats(bigwig_path=args.signal,
                                                                                     chrom_sizes_dict=chromosome_length_dictionary,
                                                                                     blacklist_path=args.blacklist,
                                                                                     max_percentile=args.max_percentile,
                                                                                     prefix=os.path.join(args.output,
                                                                                                         args.prefix))

        logging.error("Sample Statistics" +
                      "\n  Genomic minimum value: " + str(min_value) +
                      "\n  Genomic max value: " + str(max_value) +
                      "\n  Genomic median (non-zero): " + str(median) +
                      "\n  Genomic median absolute deviation (non-zero): " + str(mad) +
                      "\n  Genomic mean: " + str(mean_value) +
                      "\n  Genomic standard deviation: " + str(std_value))

    logging.error("Normalize and Write BigWig file")

    # With the file for input and output open write the header to the output file
    with pyBigWig.open(args.signal) as input_bw, pyBigWig.open(OUTPUT_FILENAME, "w") as output_bw:
        header = [(x, chromosome_length_dictionary[x]) for x in sorted(args.chroms)]

        output_bw.addHeader(header)

        # For every chromosome in header, perform normalization
        for chrom_name, chrom_length in header:
            chr_vals = np.nan_to_num(input_bw.values(chrom_name, 0, chrom_length, numpy=True))

            if args.method == "min-max":
                normalized_signal = minmax_normalize_array(chr_vals, min_value, max_value, clip=args.clip)

            elif args.method == "median-mad":
                normalized_signal = median_mad_normalize_array(chr_vals, median, mad)

            elif args.method == "zscore":
                normalized_signal = zscore_normalize_array(chr_vals, median, mad)

            elif args.method == "arcsinh":
                normalized_signal = arcsinh_normalize_array(chr_vals)

            else:
                raise NameError('Wrong normalization')

            # Import the blacklist mask for chromosome
            blacklist_mask = chromosome_blacklist_mask(args.blacklist,
                                                       chrom_name,
                                                       chromosome_length_dictionary[chrom_name]
                                                       )

            # Convert all blacklisted regions to 0
            normalized_signal[~blacklist_mask] = 0

            output_bw.addEntries(chroms=chrom_name,
                                 starts=0,
                                 ends=chrom_length,
                                 span=1,
                                 step=1,
                                 values=normalized_signal.tolist()
                                 )

    logging.error("Results saved to: " + output_dir)
