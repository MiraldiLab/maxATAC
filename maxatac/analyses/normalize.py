import logging
import timeit
import numpy as np
import pyBigWig
import os

from maxatac.utilities.system_tools import get_dir, Mute
from maxatac.utilities.genome_tools import build_chrom_sizes_dict, chromosome_blacklist_mask
from maxatac.utilities.normalization_tools import minmax_normalize_array, zscore_normalize_array, get_genomic_stats, \
    arcsinh_normalize_array


def run_normalization(args):
    """
    Normalize a bigwig file
    
    This function will normalize a bigwig array based on the desired method:

    -min-max
    -zscore
    -arcsinh

    The code will loop through each chromosome and find the min and max values. It will then create a dataframe of
    the values per chromosome. It will then scale all other values.

    Workflow Overview

    1) Create directories and set up filenames
    2) Build a dictionary of the chromosomes sizes.
    3) Find the genomic min and max values by looping through each chromosome
    5) Loop through each chromosome and minmax normalize the values based on the genomic values.

    :param args: signal, output_dir, chromosome_sizes, chromosomes, max_percentile, method, blacklist

    :return: A minmax normalized bigwig file
    """
    # Start Timer
    startTime = timeit.default_timer()

    # Set up the output directory
    output_dir = get_dir(args.output_dir)

    # Set up the output filename.
    output_filename = os.path.join(output_dir, args.name + ".bw")

    logging.info("Normalizing" +
                  "\n  Input bigwig file: " + args.signal +
                  "\n  Output filename: " + output_filename +
                  "\n  Output directory: " + output_dir +
                  "\n  Using " + args.method + " normalization"
                  )

    # Build a dictionary of chrom sizes to use to write the bigwig
    chromosome_length_dictionary = build_chrom_sizes_dict(args.chromosomes, args.chrom_sizes)

    # If the user provides a max to normalize to, use that value
    if args.max:
        logging.info("Using provided minimum and maximum values for normalization")
        logging.info("Minimum value: " + str(args.min) + "\n" + "Maximum value: " + str(args.max))
        max_value = args.max
        min_value = args.min

    else:
        logging.info("Calculating stats per chromosome")

        min_value, max_value, median, mean_value, std_value = get_genomic_stats(bigwig_path=args.signal,
                                                                                chrom_sizes_dict=chromosome_length_dictionary,
                                                                                blacklist_path=args.blacklist_bw,
                                                                                max_percentile=args.max_percentile,
                                                                                name=args.name,
                                                                                output_dir=output_dir)

        logging.info("Sample Statistics" +
                      "\n  Genomic minimum value: " + str(min_value) +
                      "\n  Genomic max value: " + str(max_value) +
                      "\n  Genomic median (non-zero): " + str(median) +
                      "\n  Genomic mean: " + str(mean_value) +
                      "\n  Genomic standard deviation: " + str(std_value))

    logging.info("Normalize and Write BigWig file")

    # With the file for input and output open write the header to the output file
    with pyBigWig.open(args.signal) as input_bw, pyBigWig.open(output_filename, "w") as output_bw:
        header = [(x, chromosome_length_dictionary[x]) for x in sorted(args.chromosomes)]

        output_bw.addHeader(header)

        # For every chromosome in header, perform normalization
        for chrom_name, chrom_length in header:
            chr_vals = np.nan_to_num(input_bw.values(chrom_name, 0, chrom_length, numpy=True))

            if args.method == "min-max":
                normalized_signal = minmax_normalize_array(chr_vals, min_value, max_value, clip=args.clip)

            elif args.method == "zscore":
                normalized_signal = zscore_normalize_array(chr_vals, mean=mean_value, std_dev=std_value)

            elif args.method == "arcsinh":
                normalized_signal = arcsinh_normalize_array(chr_vals)

            else:
                raise NameError('Wrong normalization')

            # Import the blacklist mask for chromosome
            blacklist_mask = chromosome_blacklist_mask(args.blacklist_bw,
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

    # Measure time of averaging
    stopTime = timeit.default_timer()
    totalTime = stopTime - startTime

    # Output running time in a nice format.
    mins, secs = divmod(totalTime, 60)
    hours, mins = divmod(mins, 60)

    logging.info("Total normalization time: %d:%d:%d.\n" % (hours, mins, secs))

    logging.info("Results saved to: " + output_dir)
