import logging
import pyBigWig
import os
import numpy as np
import tqdm

from maxatac.utilities.genome_tools import build_chrom_sizes_dict, GetBigWigValues
from maxatac.utilities.system_tools import get_dir


def run_averaging(args):
    """
    Average multiple bigwig files into one file

    :param args:
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

        # Create a status bar for to look fancy and count what chromosome you are on
        chrom_status_bar = tqdm.tqdm(total=len(args.chromosomes), desc='Chromosomes Processed', position=0)

        # Loop through the chromosomes and average the values across files
        for chrom_name, chrom_length in header:
            # Create an array of zeroes to start the averaging process
            chrom_vals = np.zeros(chrom_length)

            # Create a file status bar to look fancy and count what the files you are working on
            file_status_bar = tqdm.tqdm(total=len(args.bigwig_files), desc='Extracting Values from Bigwigs', position=1)

            # Loop through the bigwig files and get the values and add them to the array.
            for bigwig_file in args.bigwig_files:
                chrom_vals += GetBigWigValues(bigwig_file, chrom_name, chrom_length)

                # Update the file status bar
                file_status_bar.update(1)

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

            # Update the chromosome status bar
            chrom_status_bar.update(1)

    logging.error("Results saved to: " + output_bigwig_filename)