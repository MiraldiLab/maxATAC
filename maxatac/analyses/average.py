import logging
import pyBigWig
import os

from maxatac.utilities.system_tools import get_dir, Mute

with Mute():
    from maxatac.utilities.genome_tools import build_chrom_sizes_dict, get_bigwig_values
    import numpy as np


def run_averaging(args):
    """
    Average multiple bigwig files into one file.

    This function can take a list of input bigwig files and averages their scores using pyBigWig. The only requirement
    for the bigwig files is that they contain the same chromosomes or there might be an error about retrieving scores.
    ________________________
    Workflow Overview

    1) Create directories and set up filenames
    2) Build a dictionary of chromosome sizes and filter it based on desired chromosomes to average
    3) Open the bigwig file for writing
    4) Loop through each entry in the chromosome sizes dictionary and calculate the average across all inputs
    5) Write the bigwig file

    :param args: bigwig_files, output_dir, prefix, chromosomes, chrom_sizes

    :return: One bigwig file that is the average of input files
    """
    # Get the number of files in the input args parser
    number_input_bigwigs = len(args.bigwig_files)

    # Make the output directory
    output_dir = get_dir(args.output_dir)

    # Make the output bigwig filename from the output directory and the prefix
    output_bigwig_filename = os.path.join(output_dir, args.prefix + ".bw")

    logging.error("Averaging " + str(number_input_bigwigs) + " bigwig files. \n" +
                  "Input bigwig files: \n   - " + "\n   - ".join(args.bigwig_files) + "\n" +
                  "Output prefix: " + args.prefix + "\n" +
                  "Output directory: " + output_dir + "\n" +
                  "Output filename: " + output_bigwig_filename + "\n" +
                  "Restricting to chromosomes: \n   - " + "\n   - ".join(args.chromosomes) + "\n"
                  )

    # Build a dictionary of chromosomes sizes using the chromosomes and chromosome sizes files provided
    # The function will filter the dictionary based on the input list
    chromosome_sizes_dictionary = build_chrom_sizes_dict(args.chromosomes, args.chrom_sizes)

    # Open the bigwig file for writing
    with pyBigWig.open(output_bigwig_filename, "w") as output_bw:
        # Add a header based on the chromosomes in the chromosome sizes dictionary
        header = [(x, chromosome_sizes_dictionary[x]) for x in sorted(args.chromosomes)]

        output_bw.addHeader(header)

        # Loop through the chromosomes and average the values across files
        for chrom_name, chrom_length in header:
            # Create an array of zeroes to start the averaging process
            chrom_vals = np.zeros(chrom_length)

            # Loop through the bigwig files and get the values and add them to the array.
            for bigwig_file in args.bigwig_files:
                chrom_vals += get_bigwig_values(bigwig_file, chrom_name, chrom_length)

            # After looping through the files average the values
            chrom_vals = chrom_vals / number_input_bigwigs

            # Write the entries to the bigwig for this chromosome. Current resolution is at 1 bp.
            output_bw.addEntries(
                chroms=chrom_name,
                starts=0,
                ends=chrom_length,
                span=1,
                step=1,
                values=chrom_vals.tolist()
            )

    logging.error("Results saved to: " + output_bigwig_filename)